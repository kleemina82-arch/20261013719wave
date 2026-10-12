"""data_collection_v2.py — M-DA 기준 구현 v2 (data_collection.py 기반 확장)

파일: data_collection_v2.py | version 1.1.0 | date 2026-10-13
기준 문서: 통합 마스터 설계 v1.1 + 모듈 IO 계약 v1.12 + M-DA 최소 수정 명세 v1.1

기존 data_collection.py(업로드 원본, 수정 금지)의 구조를 계승하며 다음을 추가한다:
1. CollectedBatch 반환 — RAW 원본 + 이원 해시 + CollectionMeta
2. RAW 불변 저장 (수집 시점, M-DA 의무 — SEC-2)
3. CredentialProvider 주입 + 크리덴셜 마스킹 (SEC-1)
4. 정규화로 삭제된 봉의 식별 정보 기록 (SEC-3, TC-5.3)

품질 게이트(MIN_ROWS_RATIO 등)는 M-DC 규칙 — 본 모듈에 넣지 않는다 (책임 분리).
Revision 발급은 하류 M-UC(RevisionStore)의 몫 — 본 모듈은 배치만 제공한다.

v1.1.0 변경: RawStoreError 분리(저장 실패는 소스 폴백 대상이 아님), RawStore 배타 생성(os.link),
파일명 안전 변환(fs_safe_id), 심볼 검증 강화(선두 -/. 차단), data_basis_time UTC aware,
inf 값 정제, 폴백 이벤트를 저장 성공 후 기록.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from wave_contracts_ext import (
    CollectedBatch, CollectionMeta, CredentialProvider, EnvCredentialProvider,
    fs_safe_id, mask_credential, normalized_content_hash, raw_content_hash,
)

__version__ = "1.1.0"
VERSION_DATE = "2026-10-13"

log = logging.getLogger("data_collection_v2")


class DataSourceError(RuntimeError):
    """소스 한 곳의 수집 실패. DataCollector가 잡아서 다음 소스로 폴백한다."""


class RawStoreError(RuntimeError):
    """RAW 저장 실패. DataSourceError의 하위가 아니다 — 소스 폴백으로 가려지면 안 되며,
    DataCollector는 이 예외를 그대로 전파한다 (저장 실패 시 배치는 하류로 가지 않는다)."""


class RawStore:
    """RAW 원본 불변 저장 (M-DA 의무). append-only, 임시 파일 + 배타적 링크 생성."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(self, dataset_id: str, received_at: datetime, raw_bytes: bytes) -> str:
        path = os.path.join(
            self.base_dir,
            f"{fs_safe_id(dataset_id)}_{received_at.strftime('%Y%m%dT%H%M%S%f')}.raw")
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self.base_dir, prefix="raw_tmp_")
            with os.fdopen(fd, "wb") as f:
                f.write(raw_bytes)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, path)  # 원자적 배타 생성 — 이미 있으면 FileExistsError
            except FileExistsError:
                raise RawStoreError("RAW batch already exists — overwrite forbidden")
            except OSError:
                # 하드링크 미지원 파일시스템 대비 폴백 (경쟁 창이 남는 것은 한계로 명시)
                if os.path.exists(path):
                    raise RawStoreError("RAW batch already exists — overwrite forbidden")
                os.replace(tmp, path)
        except OSError as e:
            raise RawStoreError(f"RAW save failed: {type(e).__name__}") from e
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)
        return path


class DataSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, symbol: str, lookback_days: int = 365,
              end: Optional[date] = None) -> CollectedBatch:
        """일봉 CollectedBatch 반환. 실패 시 DataSourceError."""


def _to_utc_epoch(dates) -> List[int]:
    return [int(pd_ts.value // 10**9) if hasattr(pd_ts, "value") else int(pd_ts)
            for pd_ts in dates]


def to_collected_batch(symbol: str, timeframe: str, df, provider: str,
                       raw_bytes: bytes, source_id: str,
                       data_basis_time: Optional[datetime] = None) -> CollectedBatch:
    """정규화 + 삭제 봉 식별 기록 + CollectedBatch 구성.
    df 필수 컬럼: date(또는 DatetimeIndex), close, volume"""
    _validate_symbol(symbol)  # TC-6.2: 주입 패턴 차단 — 하드 실패
    if df is None or len(df) == 0:
        raise DataSourceError(f"{symbol}: 빈 응답")
    d = df.copy()
    if "date" in getattr(d, "columns", []):
        d = d.set_index("date")
    d["close"] = _numeric(d["close"])
    inf = [float("inf"), float("-inf")]
    d["volume"] = _numeric(d["volume"]).replace(inf, float("nan")).fillna(0.0)
    dropped: List[Dict[str, Any]] = []
    raw_rows = len(d)
    # 결측·비양수 가격 삭제 — 식별 정보 기록 (TC-5.3)
    is_nan = d["close"].isna()
    for ts in d.index[is_nan]:
        dropped.append({"timestamp": str(ts), "reason": "close NaN"})
    is_inf = d["close"].isin(inf)
    for ts in d.index[is_inf]:
        dropped.append({"timestamp": str(ts), "reason": "close non-finite"})
    is_nonpos = (~is_nan) & (~is_inf) & (d["close"] <= 0)
    for ts in d.index[is_nonpos]:
        dropped.append({"timestamp": str(ts), "reason": "close <= 0"})
    d = d[~(is_nan | is_inf | is_nonpos)]
    # 중복 인덱스 삭제 (keep=last)
    dup_mask = d.index.duplicated(keep="last")
    for ts in d.index[dup_mask]:
        dropped.append({"timestamp": str(ts), "reason": "duplicate timestamp"})
    d = d[~dup_mask].sort_index()
    if len(d) == 0:
        raise DataSourceError(f"{symbol}: 정제 후 유효 데이터 없음")

    timestamps = _to_utc_epoch(d.index)
    closes = [float(x) for x in d["close"]]
    volumes = [float(x) for x in d["volume"]]
    now = datetime.now(timezone.utc)
    if data_basis_time is None:
        last = d.index[-1]
        if hasattr(last, "to_pydatetime"):
            last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
            data_basis_time = last.to_pydatetime()  # UTC aware (received_at과 비교 가능)
        else:
            data_basis_time = now
    meta = CollectionMeta(
        source_id=source_id,
        provider=provider,
        data_basis_time=data_basis_time,
        received_at=now,
        timeframe=timeframe,
        raw_rows=raw_rows,
        normalized_rows=len(d),
        dropped=dropped,
    )
    return CollectedBatch(
        dataset_id=f"{symbol}:{timeframe}",
        raw_bytes=raw_bytes,
        raw_content_hash=raw_content_hash(raw_bytes),
        normalized_content_hash=normalized_content_hash(closes, volumes, timestamps),
        meta=meta, closes=closes, volumes=volumes, timestamps=timestamps,
    )


_SYMBOL_RE = re.compile(r"\^?[0-9A-Za-z][0-9A-Za-z.\-]*")


def _validate_symbol(symbol: str) -> None:
    """TC-6.2: 심볼 주입 차단. 형식: 선택적 선두 '^' + 영숫자로 시작 + 영숫자·'.'·'-', 최대 20자.
    선두 '-'/'.' 불가(옵션 주입·경로 우회 차단). 위반 시 하드 실패."""
    if not isinstance(symbol, str) or len(symbol) > 20 or not _SYMBOL_RE.fullmatch(symbol):
        raise DataSourceError("invalid symbol — blocked (injection guard)")


def _numeric(series):
    try:
        import pandas as pd
        return pd.to_numeric(series, errors="coerce")
    except ImportError:
        return series


class FakeJsonSource(DataSource):
    """테스트·개발용 소스 — 이미 수신된 원본 JSON 바이트로부터 배치를 구성한다.
    실제 공급자 어댑터(KIS/공공데이터포털/yfinance)는 동일 인터페이스로 추가한다.
    (실제 연동 구현은 별도 승인 범위 — 1차는 가짜 응답 주입으로 계약 검증만 수행)"""

    name = "fake_json"

    def __init__(self, provider: str = "fake", key_name: str = "FAKE_KEY",
                 credential_provider: Optional[CredentialProvider] = None):
        self.name = provider
        self.key_name = key_name
        self.cred = credential_provider or EnvCredentialProvider()

    def fetch(self, symbol: str, lookback_days: int = 365,
              end: Optional[date] = None) -> CollectedBatch:
        raise NotImplementedError("가짜 응답은 inject(raw_bytes=...) 로 주입한다")

    def inject(self, symbol: str, timeframe: str, df, raw_bytes: bytes,
               source_id: Optional[str] = None) -> CollectedBatch:
        key = self.cred.get(self.key_name)
        if not key:
            raise DataSourceError(
                f"{self.key_name} 미설정 (credential={mask_credential(key)})")
        return to_collected_batch(
            symbol, timeframe, df, provider=self.name, raw_bytes=raw_bytes,
            source_id=source_id or f"{self.name}:{symbol}")


class DataCollector:
    """소스 우선순위 + 폴백 + 최소 봉 수 확인 + 다종목."""

    def __init__(self, sources: Sequence[DataSource], min_bars: int = 30,
                 raw_store: Optional[RawStore] = None):
        if not sources:
            raise ValueError("최소 1개의 DataSource가 필요합니다.")
        self.sources = list(sources)
        self.min_bars = min_bars
        self.raw_store = raw_store
        self.last_provider: Optional[str] = None
        self.fallback_events: List[Dict[str, str]] = []

    def collect(self, symbol: str, lookback_days: int = 365,
                end: Optional[date] = None) -> CollectedBatch:
        errors: List[str] = []
        self.fallback_events = []
        for i, src in enumerate(self.sources):
            try:
                batch = src.fetch(symbol, lookback_days, end)
                if len(batch.closes) < self.min_bars:
                    raise DataSourceError(
                        f"{symbol}: {len(batch.closes)}봉 < 최소 {self.min_bars}봉")
                # RAW 불변 저장 — M-DA 의무 (SEC-2). 저장 실패(RawStoreError)는 소스 실패가
                # 아니므로 폴백하지 않고 전파한다 — 배치는 하류로 가지 않는다.
                if self.raw_store is not None:
                    self.raw_store.save(batch.dataset_id, batch.meta.received_at, batch.raw_bytes)
                if i > 0:
                    # 폴백 발생 기록 (저장 성공 후) — 직전 실패 소스 → 실제 사용 소스 (TC-5.2).
                    # 감사용 영속 근거는 Revision 이력의 provider 필드(RevisionStore)가 맡는다.
                    self.fallback_events.append({"symbol": symbol, "from": self.sources[i - 1].name,
                                                 "to": src.name})
                self.last_provider = batch.meta.provider
                log.info("%s ← %s (%d봉)", symbol, src.name, len(batch.closes))
                return batch
            except DataSourceError as e:
                log.warning("%s 실패 [%s]: %s", symbol, src.name, e)
                errors.append(f"[{src.name}] {e}")
        raise DataSourceError(f"{symbol}: 모든 소스 실패 → " + " | ".join(errors))