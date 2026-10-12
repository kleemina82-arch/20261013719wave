"""revision_store.py — RevisionStore (M-UC/Registry 측 소유, M-DA 최소 수정 명세 v1.1 제1부)

파일: revision_store.py | version 1.1.0 | date 2026-10-13
기준 문서: 통합 마스터 설계 v1.1 + 모듈 IO 계약 v1.12 + M-DA 최소 수정 명세 v1.1

소유권 확정: Revision 발급 판정은 M-UC가 본 저장소에 요청해 수행한다.
- 발급 조건(v1.1.0 확장): normalized_content_hash 상이 또는 provider 상이 → seq +1.
  둘 다 동일할 때만 no-op (revision_seq 미증가). 비교 대상은 항상 직전 Revision.
- 저장물(Revision당): 원본 payload(.raw, 불변성 입증용) + 정규화 시계열(.series.json,
  변경 분류용) + index 항목. index 기록이 커밋 지점이다.
- 덮어쓰기 금지: {dataset_id}_r{n} append-only. 이미 있는 seq는 오류 —
  단, index에 반영되지 않은 고아 payload가 같은 원본 해시이면 이어서 완료한다(크래시 복구).
- 원자성: 임시 파일 → os.replace. 쓰기 순서: payload → series → index(마지막).
- 동시성: 프로세스 내 dataset_id 단위 락 (다중 프로세스는 범위 밖).
- 변경 분류(두 축): provider_changed(bool) × data_change(NONE/DATA UPDATE/DATA REVISION).

v1.1.0 변경: 위 발급 조건·시계열 저장·두 축 분류·커밋 순서·고아 복구·파일명 안전 변환,
assert_downstream_allowed 추가. classify_change는 인스턴스 메서드로 바뀌었고
ChangeClassification을 반환한다(.label 로 기존 문자열 표현).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from wave_contracts_ext import CollectedBatch, fs_safe_id, raw_content_hash

__version__ = "1.1.0"
VERSION_DATE = "2026-10-13"

NONE = "NONE"
DATA_UPDATE = "DATA UPDATE"
DATA_REVISION = "DATA REVISION"
PROVIDER_CHANGE = "PROVIDER CHANGE"
NO_CHANGE = "impact_no_change"


@dataclass
class RevisionRecord:
    dataset_id: str
    revision_seq: int
    raw_content_hash: str
    normalized_content_hash: str
    provider: str
    received_at: str
    payload_ref: str
    series_ref: str = ""  # v1.1.0 추가. 이전 index 항목에는 없으므로 기본값 허용.


@dataclass(frozen=True)
class ChangeClassification:
    """변경 분류 — 두 축. 공급자 변경과 데이터 변경이 동시에 일어나도 둘 다 보존된다."""
    provider_changed: bool
    data_change: str  # NONE | DATA UPDATE | DATA REVISION

    @property
    def label(self) -> str:
        labels = []
        if self.provider_changed:
            labels.append(PROVIDER_CHANGE)
        if self.data_change != NONE:
            labels.append(self.data_change)
        return " + ".join(labels) if labels else NO_CHANGE


class RevisionExistsError(RuntimeError):
    """이미 존재하는 revision_seq 저장 시도 — 덮어쓰기 금지 위반."""


class ReprocessBlockedError(RuntimeError):
    """BLOCK-REPROCESS 상태에서 하류 호출 시도 — 하류 차단."""


def classify_series(prev: Dict[str, List], new: Dict[str, List]) -> str:
    """정규화 시계열 두 개의 데이터 변경 종류. 값은 정확 일치(허용오차 없음, 해시와 일관).
    - 이전 타임스탬프가 사라짐 / 겹치는 구간 값 상이 / 이전 마지막 봉보다 앞선 봉 추가(백필)
      → DATA REVISION
    - 이전 마지막 봉 이후의 신규 봉만 추가 → DATA UPDATE
    - 동일 → NONE"""
    p = {t: (c, v) for t, c, v in zip(prev["timestamps"], prev["closes"], prev["volumes"])}
    n = {t: (c, v) for t, c, v in zip(new["timestamps"], new["closes"], new["volumes"])}
    if any(t not in n for t in p):
        return DATA_REVISION
    if any(n[t] != p[t] for t in p):
        return DATA_REVISION
    added = [t for t in n if t not in p]
    if not added:
        return NONE
    if p and min(added) < max(p):
        return DATA_REVISION
    return DATA_UPDATE


class RevisionStore:
    """M-UC 측 Revision 저장소. 저장·조회·점검만 — 신뢰도·주문은 범위 밖."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._lock_guard = threading.Lock()
        # 간이 체인 상태 (Registry는 별도 구현 범위 — 본 저장소는 TC-4.x 검증용 최소 상태만 유지.
        # 메모리 전용 — 재시작 시 휘발됨. 영속화는 Registry 구현 범위.)
        self._chain_state: Dict[str, str] = {}      # dataset_id -> "ACTIVE"|"INTERRUPTED"|"BLOCK-REPROCESS"
        self._last_result: Dict[str, RevisionRecord] = {}  # 마지막 유효 결과 (역사적 유효성 보존)

    def _lock(self, dataset_id: str) -> threading.Lock:
        with self._lock_guard:
            if dataset_id not in self._locks:
                self._locks[dataset_id] = threading.Lock()
            return self._locks[dataset_id]

    # ---------------- 조회 ----------------
    def latest(self, dataset_id: str) -> Optional[RevisionRecord]:
        recs = self._records(dataset_id)
        return recs[-1] if recs else None

    def get(self, dataset_id: str, revision_seq: int) -> Optional[RevisionRecord]:
        for r in self._records(dataset_id):
            if r.revision_seq == revision_seq:
                return r
        return None

    def _records(self, dataset_id: str) -> List[RevisionRecord]:
        index_path = self._index_path(dataset_id)
        if not os.path.exists(index_path):
            return []
        with open(index_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        return [RevisionRecord(**e) for e in entries]

    def load_series(self, dataset_id: str, revision_seq: int) -> Optional[Dict[str, List]]:
        """해당 Revision의 정규화 시계열. 없으면(구버전 항목 등) None."""
        rec = self.get(dataset_id, revision_seq)
        if rec is None or not rec.series_ref or not os.path.exists(rec.series_ref):
            return None
        with open(rec.series_ref, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------- 저장 (M-UC 호출) ----------------
    def save(self, batch: CollectedBatch) -> RevisionRecord:
        """발급 판정: 정규화 해시와 provider가 모두 직전 Revision과 동일 → no-op,
        하나라도 상이 → 신규 발급."""
        with self._lock(batch.dataset_id):
            latest = self.latest(batch.dataset_id)
            if (latest is not None
                    and latest.normalized_content_hash == batch.normalized_content_hash
                    and latest.provider == batch.meta.provider):
                return latest  # 멱등 no-op — Revision은 증가하지 않는다

            new_seq = (latest.revision_seq + 1) if latest else 1
            payload_ref = self._payload_path(batch.dataset_id, new_seq)
            series_ref = self._series_path(batch.dataset_id, new_seq)
            if os.path.exists(payload_ref):
                # index에 없는 seq의 payload = 이전 시도가 index 기록 전에 중단된 고아.
                # 같은 원본이면 이어서 완료(재기록은 동일 내용), 다르면 덮어쓰기 금지 위반.
                with open(payload_ref, "rb") as f:
                    orphan_hash = raw_content_hash(f.read())
                if orphan_hash != batch.raw_content_hash:
                    raise RevisionExistsError(
                        f"revision {new_seq} already exists — overwrite forbidden")
            record = RevisionRecord(
                dataset_id=batch.dataset_id,
                revision_seq=new_seq,
                raw_content_hash=batch.raw_content_hash,
                normalized_content_hash=batch.normalized_content_hash,
                provider=batch.meta.provider,
                received_at=batch.meta.received_at.isoformat(),
                payload_ref=payload_ref,
                series_ref=series_ref,
            )
            self._commit(batch, record)
            return record

    # ---------------- 원자적 쓰기 (TC-1.3/1.4) ----------------
    @staticmethod
    def _atomic_write_bytes(path: str, data: bytes) -> None:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _write_index(self, index_path: str, entries: List[Dict[str, Any]]) -> None:
        """index 갱신(커밋 지점) — 임시파일 + 원자적 rename."""
        self._atomic_write_bytes(index_path, json.dumps(entries).encode("utf-8"))

    def _commit(self, batch: CollectedBatch, record: RevisionRecord) -> None:
        """쓰기 순서: payload → series → index(마지막). index에 반영되기 전까지 Revision은
        존재하지 않는다. 중간 실패 시 남는 것은 고아 payload/series뿐이며 재시도가 복구한다."""
        self._atomic_write_bytes(record.payload_ref, batch.raw_bytes)
        series = {"timestamps": list(batch.timestamps),
                  "closes": list(batch.closes),
                  "volumes": list(batch.volumes)}
        self._atomic_write_bytes(record.series_ref, json.dumps(series).encode("utf-8"))
        entries = [r.__dict__ for r in self._records(record.dataset_id)]
        entries.append(record.__dict__)
        self._write_index(self._index_path(record.dataset_id), entries)

    # ---------------- 무결성 점검 (TC-1.1) ----------------
    def verify_payload(self, dataset_id: str, revision_seq: int) -> bool:
        """동일 저장 배치의 원본 바이트와 재계산 해시가 일치하는지."""
        rec = self.get(dataset_id, revision_seq)
        if rec is None:
            return False
        with open(rec.payload_ref, "rb") as f:
            data = f.read()
        return raw_content_hash(data) == rec.raw_content_hash

    # ---------------- 변경 원인 분류 (M-UC-2, TC-3.x) ----------------
    def classify_change(self, prev: Optional[RevisionRecord],
                        new: RevisionRecord) -> ChangeClassification:
        """두 축 분류. 같은 Revision(no-op 반환)이면 변경 없음.
        시계열이 없는 구버전 항목은 해시 비교만 가능 → 해시 상이 시 보수적으로 DATA REVISION."""
        if prev is None:
            return ChangeClassification(False, DATA_UPDATE)
        if prev.dataset_id == new.dataset_id and prev.revision_seq == new.revision_seq:
            return ChangeClassification(False, NONE)
        provider_changed = prev.provider != new.provider
        p = self.load_series(prev.dataset_id, prev.revision_seq)
        n = self.load_series(new.dataset_id, new.revision_seq)
        if p is not None and n is not None:
            data_change = classify_series(p, n)
        elif prev.normalized_content_hash != new.normalized_content_hash:
            data_change = DATA_REVISION
        else:
            data_change = NONE
        return ChangeClassification(provider_changed, data_change)

    # ---------------- 장애·차단 상태 (TC-4.x) ----------------
    def mark_interrupted(self, dataset_id: str) -> None:
        self._chain_state[dataset_id] = "INTERRUPTED"

    def mark_block_reprocess(self, dataset_id: str) -> None:
        self._chain_state[dataset_id] = "BLOCK-REPROCESS"

    def mark_active(self, dataset_id: str) -> None:
        self._chain_state[dataset_id] = "ACTIVE"

    def downstream_allowed(self, dataset_id: str) -> bool:
        """BLOCK-REPROCESS 상태에서는 하류 호출 불가 (TC-4.3)."""
        return self._chain_state.get(dataset_id) != "BLOCK-REPROCESS"

    def assert_downstream_allowed(self, dataset_id: str) -> None:
        """하류 호출 직전 게이트. 차단 상태면 ReprocessBlockedError."""
        if not self.downstream_allowed(dataset_id):
            raise ReprocessBlockedError("BLOCK-REPROCESS — 하류 호출 차단")

    def record_last_valid(self, dataset_id: str, record: RevisionRecord) -> None:
        """역사적 유효성 보존 — 재검증 실패 시에도 과거 판정 기록은 삭제되지 않는다 (TC-4.1)."""
        self._last_result[dataset_id] = record

    def last_valid(self, dataset_id: str) -> Optional[RevisionRecord]:
        return self._last_result.get(dataset_id)

    # ---------------- 경로 ----------------
    def _index_path(self, dataset_id: str) -> str:
        return os.path.join(self.base_dir, f"{fs_safe_id(dataset_id)}_index.json")

    def _payload_path(self, dataset_id: str, seq: int) -> str:
        return os.path.join(self.base_dir, f"{fs_safe_id(dataset_id)}_r{seq}.raw")

    def _series_path(self, dataset_id: str, seq: int) -> str:
        return os.path.join(self.base_dir, f"{fs_safe_id(dataset_id)}_r{seq}.series.json")
