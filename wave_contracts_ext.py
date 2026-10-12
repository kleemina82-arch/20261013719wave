"""wave_contracts_ext.py — 계약 데이터 구조 + CredentialProvider (M-DA 최소 수정 명세 v1.1 기반)

파일: wave_contracts_ext.py | version 1.1.0 | date 2026-10-13
기준 문서: 통합 마스터 설계 v1.1 + 모듈 IO 계약 v1.12 + M-DA 최소 수정 명세 v1.1 + 계약 테스트 케이스 v1.1(확정 28개 중 1차 24개 대상)

책임 경계 (명세 v1.1 확정):
- revision 발급 주체는 M-UC(RevisionStore) — 본 모듈은 발급하지 않는다.
- rule_set_version은 M-DC 검증 시 부여 — 본 모듈이 채우지 않는다.
- 해시 재현성 규칙: 테스트 케이스 0-1 (열 순서 close→volume→timestamp, 오름차순 정렬,
  UTC 정규화, float64 little-endian 바이트).

v1.1.0 변경: 해시 직렬화를 명세 0-1(열 단위 close→volume→timestamp, 전부 float64 LE)로 정정,
fs_safe_id 추가(파일명 안전 변환). 해시값이 v1.0.0과 달라진다 — 실사용 전 정정.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from urllib.parse import quote
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__version__ = "1.1.0"
VERSION_DATE = "2026-10-13"


# ----------------------------------------------------------------
# CredentialProvider — SEC-1 (명세 v1.1 제3부)
# ----------------------------------------------------------------
@runtime_checkable
class CredentialProvider(Protocol):
    """자격증명 접근 인터페이스. 기본 구현은 환경변수(기존 동작 호환).
    외부 Secret Store는 이 프로토콜 뒤에 교체한다."""

    def get(self, name: str) -> Optional[str]: ...


class EnvCredentialProvider:
    """기본 구현: 환경변수. 값은 절대 로그/메타데이터/예외 메시지에 노출하지 않는다."""

    def get(self, name: str) -> Optional[str]:
        return os.getenv(name)


def mask_credential(value: Optional[str]) -> str:
    """예외·로그용 마스킹. 값 전체 또는 부분문자열을 절대 반환하지 않는다."""
    if not value:
        return "<unset>"
    return f"***({len(value)} chars)"


def assert_no_credential_leak(text: str, secrets: List[str]) -> None:
    """TC-6.1/TC-6.3 지원: text에 secrets의 부분문자열이 있으면 AssertionError."""
    for s in secrets:
        if s and s in text:
            raise AssertionError("credential leak detected in output text")


# ----------------------------------------------------------------
# CollectionMeta — SEC-3 / V-1 provenance (명세 v1.1 제2부)
# 단계별 필수 필드: RAW 단계에서 revision=None, rule_set_version=None 이 정상.
# ----------------------------------------------------------------
@dataclass
class CollectionMeta:
    source_id: str                      # 예: "kis:005930:20260101-20261009"
    provider: str                       # "kis" / "data.go.kr" / "yfinance" / ...
    data_basis_time: datetime           # 데이터 기준 시각
    received_at: datetime               # 수집 완료 시각
    timeframe: str = "1d"
    revision: Optional[int] = None      # M-UC 발급 후 기록. M-DA 생성 시점엔 None.
    rule_set_version: Optional[str] = None  # M-DC 검증 시 부여. M-DA는 채우지 않는다.
    raw_rows: int = 0
    normalized_rows: int = 0
    dropped: List[Dict[str, Any]] = field(default_factory=list)  # TC-5.3: 삭제 봉 식별 정보

    def required_fields_present(self) -> bool:
        """RAW 수집 단계 필수 필드 검증 (테스트 케이스 0-2 표 기준)."""
        return bool(self.source_id) and self.data_basis_time is not None and self.received_at is not None


# ----------------------------------------------------------------
# 해시 — 이원 해시 (명세 v1.1 확정)
# ----------------------------------------------------------------
def normalized_content_hash(closes, volumes, timestamps) -> str:
    """정규화 데이터 해시 — Revision 비교·발급 판정용.
    규칙(테스트 케이스 0-1): 열 단위(column-major) 직렬화, 순서 close→volume→timestamp,
    timestamp 오름차순 정렬(세 열 동일 순서), timestamp는 UTC epoch 초,
    모든 값은 float64 little-endian 바이트.
    입력 계약: 세 열 길이 동일, timestamp 중복 없음, 모든 값 유한(NaN/inf 불가)
    — 위반 시 ValueError (상류 M-DA가 정제해서 넘겨야 한다)."""
    n = len(timestamps)
    if not (len(closes) == len(volumes) == n):
        raise ValueError("closes/volumes/timestamps length mismatch")
    order = sorted(range(n), key=lambda i: timestamps[i])
    for a, b in zip(order, order[1:]):
        if timestamps[a] == timestamps[b]:
            raise ValueError("duplicate timestamp in hash input")

    def col(values):
        out = []
        for i in order:
            v = float(values[i])
            if not math.isfinite(v):
                raise ValueError("non-finite value in hash input")
            out.append(v + 0.0)  # -0.0 → 0.0 정규화
        return struct.pack(f"<{n}d", *out)

    h = hashlib.sha256()
    h.update(col(closes))
    h.update(col(volumes))
    h.update(col(timestamps))
    return h.hexdigest()


def fs_safe_id(dataset_id: str) -> str:
    """dataset_id → 파일명 안전 문자열. ':' '^' 등을 퍼센트 인코딩(단사)한다.
    윈도우(NTFS)에서 ':'가 대체 데이터 스트림으로 해석되는 문제를 막는다."""
    return quote(dataset_id, safe="")


def raw_content_hash(raw_bytes: bytes) -> str:
    """RAW 불변성 입증용 — 공급자 원본 바이트 그대로 (직렬화 규칙 미적용)."""
    return hashlib.sha256(raw_bytes).hexdigest()


# ----------------------------------------------------------------
# CollectedBatch — M-DA가 하류(M-UC)에 넘기는 배치 (명세 v1.1 제1-3(b))
# ----------------------------------------------------------------
@dataclass
class CollectedBatch:
    dataset_id: str                      # 예: "005930:1d"
    raw_bytes: bytes                     # 공급자 원본 (불변 저장본)
    raw_content_hash: str                # raw_content_hash(raw_bytes)
    normalized_content_hash: str
    meta: CollectionMeta
    closes: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)  # UTC epoch 초

    def __post_init__(self) -> None:
        if self.raw_content_hash != raw_content_hash(self.raw_bytes):
            raise ValueError("raw_content_hash mismatch — RAW 불변성 위반")


# ----------------------------------------------------------------
# 신뢰도 상한 — M-PC C_final = min(C_raw, C_max) (IO 계약 v1.7, TC-7.x 대상)
# 본 모듈은 상한 적용·하드 거부 '규칙'만 제공한다. 산식(C_raw)은 M-PC 구현 범위.
# ----------------------------------------------------------------
class HardRejectError(RuntimeError):
    """V-9 하드 거부 — 상한 초과 최종 출력은 우회 불가."""


def apply_confidence_cap(c_raw: float, c_max: float) -> float:
    """TC-7.1: 상한 적용 검증. 상한 미적용 값을 출력하면 안 된다."""
    if c_raw > c_max:
        return c_max
    return c_raw


def assert_final_output_within_cap(c_final: float, c_max: float) -> None:
    """TC-7.2: 최종 출력이 상한을 초과하면 하드 거부 (조건부·점수생략 우회 불가)."""
    if c_final > c_max:
        raise HardRejectError(f"V-9: final confidence {c_final} exceeds cap {c_max}")


def score_unverified_input(meta: CollectionMeta) -> float:
    """TC-7.3: provenance 불완전 입력에는 점수를 부여하지 않고 차단.
    검증되지 않은 입력은 계산 자체를 수행하지 않는다."""
    if not meta.required_fields_present():
        raise HardRejectError("unverified input — provenance incomplete, scoring blocked")
    raise NotImplementedError(
        "C_raw scoring is M-PC implementation scope — 1차 범위에서는 미구현 (가짜 통과 금지)"
    )