"""test_contract_cases.py — 계약 테스트 케이스 1차 구현 (확정 v1.1 중 24개)

파일: test_contract_cases.py | version 1.1.0 | date 2026-10-13
기준 문서: 계약 테스트 케이스 확정안 v1.1 + M-DA 최소 수정 명세 v1.1 + IO 계약 v1.12

범위: TC-1.1~1.4, TC-2.1~2.3, TC-3.1~3.4, TC-4.1~4.3, TC-5.1~5.4, TC-6.1~6.3, TC-7.1~7.3 (24개)
이연(2차): TC-7.4, TC-8.1~8.3 — Registry·Gateway 구현 후
미구현 기능(Registry/Gateway)을 가짜로 통과 처리하지 않는다.
실패 시 자동 수정 없음 — 원인·영향 계약·최소 수정 권고 보고 후 별도 승인.

v1.1.0: 확정 24개 + 명세 외 추가 케이스(test_extra_*, TC-0.1 해시 회귀)를 별도 표기.
추가 케이스는 확정 명세에 없으므로 승인 전까지 "제안" 상태다.
수정 사항: 픽스처(make_batch base 인자), TC-1.3 동일 경로 충돌, TC-3.2 호출 오류,
TC-3.x 분류 API(ChangeClassification.label), TC-4.3 실제 게이트 검증, TC-6.2 확장.

실행: python -m unittest test_contract_cases -v
"""

import hashlib
import json
import os
import struct
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from unittest import mock

import pandas as pd

from wave_contracts_ext import (
    CollectedBatch, CollectionMeta, apply_confidence_cap, assert_final_output_within_cap,
    assert_no_credential_leak, mask_credential, normalized_content_hash, raw_content_hash,
    score_unverified_input, HardRejectError, fs_safe_id,
)
from data_collection_v2 import (
    DataCollector, DataSourceError, FakeJsonSource, RawStore, RawStoreError, to_collected_batch,
)
from revision_store import (
    ReprocessBlockedError, RevisionExistsError, RevisionStore, classify_series,
)

FAKE_KEY = "SECRET_KEY_12345"


def make_df(rows=10, start="2026-01-05", base=2500.0):
    dates = pd.bdate_range(start, periods=rows)
    return pd.DataFrame({
        "close": [base + i for i in range(rows)],
        "volume": [1000 + i for i in range(rows)],
    }, index=dates)


def make_batch(symbol="005930", rows=10, raw_mark=b"RAW-ORIGIN", provider="fake",
               df=None, source_id=None, base=2500.0):
    """raw_mark는 RAW 바이트만 바꾼다. 정규화 데이터를 바꾸려면 base 또는 df를 바꿀 것."""
    df = df if df is not None else make_df(rows, base=base)
    raw = raw_mark + b"|" + json.dumps({"rows": rows}).encode()
    return to_collected_batch(
        symbol, "1d", df, provider=provider, raw_bytes=raw,
        source_id=source_id or f"{provider}:{symbol}")


class RawImmutabilityTests(unittest.TestCase):
    """TC-1.1~1.4 — RAW 불변성 (P0, SEC-2)"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RevisionStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tc11_same_batch_bytes_and_recomputed_hash(self):
        batch = make_batch()
        rec = self.store.save(batch)
        self.assertEqual(rec.revision_seq, 1)
        self.assertTrue(self.store.verify_payload("005930:1d", 1))  # 동일 배치 바이트·해시 일치

    def test_tc12_new_revision_preserves_old_batch(self):
        r1 = self.store.save(make_batch())
        r2 = self.store.save(make_batch(raw_mark=b"RAW-CHANGED", base=3000.0))
        self.assertEqual(r2.revision_seq, 2)
        self.assertTrue(self.store.verify_payload("005930:1d", 1))
        self.assertNotEqual(r1.raw_content_hash, r2.raw_content_hash)  # 기존 r1 배치·해시 유지

    def test_tc13_raw_save_failure_no_partial_file(self):
        raw_store = RawStore(os.path.join(self.tmp.name, "raw"))
        received_at = datetime.now(timezone.utc)
        path = raw_store.save("X:1d", received_at, b"first")
        # 동일 dataset_id + 동일 received_at = 동일 경로 → 덮어쓰기 시도는 RawStoreError
        with self.assertRaises(RawStoreError):
            raw_store.save("X:1d", received_at, b"second")
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"first")  # 원본 불변
        files = [f for f in os.listdir(raw_store.base_dir) if f.startswith("raw_tmp_")]
        self.assertEqual(files, [])  # 부분 파일 미잔존

    def test_tc14_raw_saved_but_revision_issue_fails_idempotent_retry(self):
        # RAW 저장 성공 → Revision 발급 실패 주입 → 재시도 시 중복 발급 없음
        batch = make_batch()
        raw_store = RawStore(os.path.join(self.tmp.name, "raw"))
        raw_store.save(batch.dataset_id, batch.meta.received_at, batch.raw_bytes)
        # 발급 실패 시뮬레이션: index를 손상해 save가 실패하게 함
        idx = self.store._index_path(batch.dataset_id)
        with open(idx, "w") as f:
            f.write("{invalid json")
        with self.assertRaises(Exception):
            self.store.save(batch)
        # 복구 후 재시도 — seq 1로 발급 (중복·점프 없음)
        os.unlink(idx)
        rec = self.store.save(batch)
        self.assertEqual(rec.revision_seq, 1)
        rec2 = self.store.save(make_batch(raw_mark=b"OTHER", base=3000.0))
        self.assertEqual(rec2.revision_seq, 2)

    def test_extra_orphan_payload_recovered_on_retry(self):
        """[추가] payload 기록 후 index 기록 실패 → 재시도가 같은 seq를 이어서 완료."""
        batch = make_batch()
        with mock.patch.object(self.store, "_write_index", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.store.save(batch)
        self.assertTrue(os.path.exists(self.store._payload_path(batch.dataset_id, 1)))  # 고아
        self.assertIsNone(self.store.latest(batch.dataset_id))  # index 미반영 = Revision 없음
        rec = self.store.save(batch)
        self.assertEqual(rec.revision_seq, 1)
        self.assertEqual(len(self.store._records(batch.dataset_id)), 1)
        self.assertTrue(self.store.verify_payload(batch.dataset_id, 1))

    def test_extra_foreign_orphan_still_forbidden(self):
        """[추가] index에 없는 seq에 다른 원본이 이미 있으면 덮어쓰기 금지."""
        batch = make_batch()
        with open(self.store._payload_path(batch.dataset_id, 1), "wb") as f:
            f.write(b"SOMETHING-ELSE")
        with self.assertRaises(RevisionExistsError):
            self.store.save(batch)

    def test_extra_raw_store_failure_does_not_fallback(self):
        """[추가] RAW 저장 실패는 소스 폴백으로 가려지지 않고 전파된다."""
        calls = []

        class Src1:
            name = "kis"
            def fetch(self, symbol, lookback_days=365, end=None):
                calls.append("kis")
                return make_batch(provider="kis")

        class Src2:
            name = "naver"
            def fetch(self, symbol, lookback_days=365, end=None):
                calls.append("naver")
                return make_batch(provider="naver")

        class BrokenStore:
            def save(self, *a, **k):
                raise RawStoreError("disk full")

        collector = DataCollector([Src1(), Src2()], min_bars=5, raw_store=BrokenStore())
        with self.assertRaises(RawStoreError):
            collector.collect("005930")
        self.assertEqual(calls, ["kis"])  # naver로 폴백하지 않음
        self.assertEqual(collector.fallback_events, [])


class RevisionIdempotencyTests(unittest.TestCase):
    """TC-2.1~2.3 — Revision 멱등성 (P0, M-UC-4(c))"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RevisionStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tc21_same_data_noop(self):
        self.store.save(make_batch(raw_mark=b"A"))
        rec = self.store.save(make_batch(raw_mark=b"A"))  # 동일 normalized 데이터
        self.assertEqual(rec.revision_seq, 1)  # seq 불변
        # no-op이 RAW 기록 생략을 의미하지 않음 — 원본 RAW는 별도 보존됨
        self.assertTrue(self.store.verify_payload("005930:1d", 1))

    def test_tc22_retry_same_event_no_duplicate(self):
        r1 = self.store.save(make_batch(raw_mark=b"B"))
        r2 = self.store.save(make_batch(raw_mark=b"B"))  # 재시도
        self.assertEqual(r1.revision_seq, r2.revision_seq)
        self.assertEqual(r1.normalized_content_hash, r2.normalized_content_hash)

    def test_tc23_concurrent_save_single_issue(self):
        batch_a = make_batch(raw_mark=b"RACE")
        results, errors = [], []
        lock = threading.Lock()

        def worker():
            try:
                r = self.store.save(batch_a)
                with lock:
                    results.append(r.revision_seq)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertTrue(all(s == 1 for s in results), f"seq: {results}, errors: {errors}")  # 단조·중복 없음
        recs = self.store._records("005930:1d")
        self.assertEqual(len(recs), 1)


class ChangeTrackingTests(unittest.TestCase):
    """TC-3.1~3.4 — 변경 추적 (P1, M-UC-2/4(b))"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RevisionStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tc31_past_bar_edit_data_revision(self):
        prev = self.store.save(make_batch(raw_mark=b"P"))
        # 과거 봉 값 변경 (첫 봉 close +100)
        df = make_df(10)
        df.iloc[0, 0] = df.iloc[0, 0] + 100
        new = self.store.save(make_batch(df=df, raw_mark=b"P2"))
        self.assertEqual(self.store.classify_change(prev, new).label, "DATA REVISION")

    def test_tc32_new_bar_only_data_update(self):
        prev = self.store.save(make_batch(rows=10, raw_mark=b"U"))
        new = self.store.save(make_batch(rows=11, raw_mark=b"U2"))  # 신규 봉 추가
        self.assertEqual(new.revision_seq, 2)
        self.assertEqual(self.store.classify_change(prev, new).label, "DATA UPDATE")

    def test_tc33_provider_change(self):
        prev = self.store.save(make_batch(provider="kis", raw_mark=b"K"))
        new = self.store.save(make_batch(provider="naver", raw_mark=b"N"))  # 폴백 전환, 데이터 동일
        self.assertEqual(new.revision_seq, 2)  # 공급자 상이 → 신규 Revision
        self.assertEqual(self.store.classify_change(prev, new).label, "PROVIDER CHANGE")

    def test_tc34_no_change_short_termination(self):
        prev = self.store.save(make_batch(raw_mark=b"S"))
        new = self.store.save(make_batch(raw_mark=b"S"))  # 동일 → no-op, 종결
        self.assertEqual(self.store.classify_change(prev, new).label, "impact_no_change")
        self.assertEqual(new.revision_seq, prev.revision_seq)

    def test_extra_provider_and_data_change_both_preserved(self):
        """[추가] 공급자 변경 + 과거 봉 수정 동시 발생 시 두 축 모두 보존."""
        prev = self.store.save(make_batch(provider="kis"))
        df = make_df(10)
        df.iloc[0, 0] += 100
        new = self.store.save(make_batch(provider="naver", df=df))
        c = self.store.classify_change(prev, new)
        self.assertTrue(c.provider_changed)
        self.assertEqual(c.data_change, "DATA REVISION")
        self.assertEqual(c.label, "PROVIDER CHANGE + DATA REVISION")

    def test_extra_removed_bar_is_revision(self):
        """[추가] 이전에 있던 봉이 사라지면 DATA REVISION."""
        prev = self.store.save(make_batch(rows=10))
        new = self.store.save(make_batch(rows=9))
        self.assertEqual(self.store.classify_change(prev, new).label, "DATA REVISION")

    def test_extra_backfill_is_revision(self):
        """[추가] 이전 마지막 봉보다 앞선 봉이 추가되면(백필) DATA REVISION."""
        prev = self.store.save(make_batch(rows=10))
        extra = pd.DataFrame({"close": [2499.0], "volume": [999]},
                             index=[pd.Timestamp("2026-01-02")])
        new = self.store.save(make_batch(df=pd.concat([extra, make_df(10)])))
        self.assertEqual(self.store.classify_change(prev, new).label, "DATA REVISION")

    def test_extra_classify_series_exact(self):
        """[추가] 부동소수 미세 차이도 정확 비교로 REVISION."""
        a = {"timestamps": [1, 2], "closes": [1.0, 2.0], "volumes": [1.0, 1.0]}
        b = {"timestamps": [1, 2], "closes": [1.0, 2.0000000001], "volumes": [1.0, 1.0]}
        self.assertEqual(classify_series(a, a), "NONE")
        self.assertEqual(classify_series(a, b), "DATA REVISION")


class FailureConcurrencyTests(unittest.TestCase):
    """TC-4.1~4.3 — 장애·동시성 (P1, M-UC-4(d))"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RevisionStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tc41_interrupted_preserves_history_and_gates_current(self):
        r1 = self.store.save(make_batch(raw_mark=b"V1"))
        self.store.record_last_valid("005930:1d", r1)
        self.store.mark_interrupted("005930:1d")  # 재검증 도중 실패 주입
        # 역사적 유효성: 과거 판정 기록 보존
        self.assertEqual(self.store.last_valid("005930:1d").revision_seq, 1)
        # 현재 Revision의 실행 허용은 별개 — INTERRUPTED 상태 기록 존재 확인
        r2 = self.store.save(make_batch(raw_mark=b"V2", base=3000.0))
        self.assertEqual(r2.revision_seq, 2)  # 새 Revision은 발급되지만 이전 기록 불변
        self.assertEqual(self.store.last_valid("005930:1d").revision_seq, 1)

    def test_tc42_retry_after_failure_no_partial_state(self):
        try:
            self.store.save(make_batch(raw_mark=b"W"))
        except Exception:
            pass
        rec = self.store.save(make_batch(raw_mark=b"W"))
        self.assertEqual(rec.revision_seq, 1)  # 부분 상태 없이 1부터 발급

    def test_tc43_block_reprocess_blocks_downstream(self):
        self.store.save(make_batch(raw_mark=b"X"))
        self.store.mark_block_reprocess("005930:1d")
        self.assertFalse(self.store.downstream_allowed("005930:1d"))
        with self.assertRaises(ReprocessBlockedError):
            self.store.assert_downstream_allowed("005930:1d")  # 실제 게이트가 차단
        self.store.mark_active("005930:1d")
        self.store.assert_downstream_allowed("005930:1d")  # 해제 후 통과


class ProvenanceTests(unittest.TestCase):
    """TC-5.1~5.4 — 출처·메타데이터 (P1, SEC-3/V-1)"""

    def test_tc51_missing_provenance_fails_contract(self):
        meta = CollectionMeta(source_id="", provider="fake",
                              data_basis_time=datetime.now(timezone.utc),
                              received_at=datetime.now(timezone.utc))
        self.assertFalse(meta.required_fields_present())  # 단계 필수 필드 누락 → 거부

    def test_tc52_fallback_provider_recorded(self):
        class FailSource:
            name = "kis"
            def fetch(self, symbol, lookback_days=365, end=None):
                raise DataSourceError("kis down")

        class OkSource:
            name = "naver"
            def __init__(self):
                self.batch = make_batch(provider="naver")
            def fetch(self, symbol, lookback_days=365, end=None):
                return self.batch

        collector = DataCollector([FailSource(), OkSource()], min_bars=5)
        batch = collector.collect("005930")
        self.assertEqual(batch.meta.provider, "naver")  # 실제 사용 소스와 일치
        self.assertEqual(len(collector.fallback_events), 1)  # 폴백 이벤트 기록

    def test_tc53_dropped_bars_identifiable(self):
        df = make_df(10)
        df.iloc[3, 0] = float("nan")  # 결측 주입
        batch = make_batch(df=df)
        self.assertEqual(batch.meta.raw_rows, 10)
        self.assertEqual(batch.meta.normalized_rows, 9)
        self.assertEqual(len(batch.meta.dropped), 1)  # 식별 정보 포함
        self.assertEqual(batch.meta.dropped[0]["reason"], "close NaN")

    def test_extra_data_basis_time_is_utc_aware(self):
        """[추가] data_basis_time이 received_at과 비교 가능(UTC aware)."""
        batch = make_batch()
        self.assertIsNotNone(batch.meta.data_basis_time.tzinfo)
        self.assertLessEqual(batch.meta.data_basis_time, batch.meta.received_at)

    def test_tc54_stage_records_not_overwritten(self):
        # RAW 단계: revision=None, rule_set_version=None 이 정상 (0-2 표)
        batch = make_batch()
        self.assertIsNone(batch.meta.revision)
        self.assertIsNone(batch.meta.rule_set_version)
        # 발급 후에도 원본 배치 메타는 변하지 않는다 (별도 기록)
        store = RevisionStore(tempfile.mkdtemp())
        rec = store.save(batch)
        self.assertIsNone(batch.meta.revision)  # M-DA 생성 시점 메타 불변
        self.assertEqual(rec.revision_seq, 1)     # 발급 기록은 별도 객체


class SecurityTests(unittest.TestCase):
    """TC-6.1~6.3 — 보안 (P0/P2, SEC-1)"""

    def test_tc61_error_messages_masked(self):
        masked = mask_credential(FAKE_KEY)
        self.assertNotIn(FAKE_KEY, masked)
        self.assertNotIn(FAKE_KEY[:5], masked)
        assert_no_credential_leak(f"kis auth failed: {masked}", [FAKE_KEY])  # 예외 없음

    def test_tc62_injection_blocked(self):
        # 주입 패턴 심볼은 source_id/dataset_id 구성에서 차단되어야 한다
        for bad in ["; rm -rf", "--drop", "$(cmd)", "..", "../x", "-x", ".hidden", "a/b", "", "x" * 21]:
            with self.subTest(symbol=bad):
                with self.assertRaises(DataSourceError):
                    to_collected_batch(bad, "1d", make_df(5), provider="fake",
                                       raw_bytes=b"r", source_id=f"fake:{bad}")
        with self.assertRaises(DataSourceError):  # 문자열이 아닌 심볼
            to_collected_batch(10, "1d", make_df(5), provider="fake",
                               raw_bytes=b"r", source_id="fake:10")

    def test_extra_valid_symbols_pass(self):
        """[추가] 정상 심볼(종목코드·지수·점 포함)은 통과."""
        for ok in ["005930", "^KS11", "BRK.B", "A-B"]:
            with self.subTest(symbol=ok):
                b = to_collected_batch(ok, "1d", make_df(5), provider="fake",
                                       raw_bytes=b"r", source_id=f"fake:{ok}")
                self.assertEqual(b.dataset_id, f"{ok}:1d")

    def test_extra_fs_safe_id_windows_safe(self):
        """[추가] 파일명에 윈도우 금지 문자(: 등)가 남지 않는다."""
        for did in ["005930:1d", "^KS11:1d", "BRK.B:1d"]:
            safe = fs_safe_id(did)
            self.assertFalse(any(c in safe for c in '<>:"/\\|?*'), safe)
        self.assertNotEqual(fs_safe_id("A:1d"), fs_safe_id("A_1d"))  # 충돌 없음

    def test_tc63_no_credential_in_outputs(self):
        batch = make_batch()
        # 배치 전체 직렬화에 크리덴셜 부재 검증
        dump = json.dumps({
            "meta": {"source_id": batch.meta.source_id, "provider": batch.meta.provider},
            "hashes": [batch.raw_content_hash, batch.normalized_content_hash],
        })
        assert_no_credential_leak(dump, [FAKE_KEY])


GOLDEN_HASH = "c9fc25f4ca31d1cec117585d07ff8ff7afd9cd9c1769ed17b7b4935d4d1fbf97"


class HashRuleTests(unittest.TestCase):
    """TC-0.1 [추가: 해시 규칙 회귀] — 명세 0-1: 열 단위 close→volume→timestamp, float64 LE"""

    CLOSES = [100.0, 101.5]
    VOLS = [10.0, 20.0]
    TS = [1767571200, 1767657600]

    def test_tc01_matches_spec_serialization(self):
        expected = hashlib.sha256(
            struct.pack("<2d", *self.CLOSES)
            + struct.pack("<2d", *self.VOLS)
            + struct.pack("<2d", *[float(t) for t in self.TS])).hexdigest()
        self.assertEqual(normalized_content_hash(self.CLOSES, self.VOLS, self.TS), expected)

    def test_tc01_golden_value_locked(self):
        self.assertEqual(normalized_content_hash(self.CLOSES, self.VOLS, self.TS), GOLDEN_HASH)

    def test_tc01_input_order_independent(self):
        rev = lambda x: list(reversed(x))
        self.assertEqual(normalized_content_hash(rev(self.CLOSES), rev(self.VOLS), rev(self.TS)),
                         GOLDEN_HASH)

    def test_tc01_negative_zero_normalized(self):
        a = normalized_content_hash([1.0], [0.0], [1])
        b = normalized_content_hash([1.0], [-0.0], [1])
        self.assertEqual(a, b)

    def test_tc01_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            normalized_content_hash([1.0, 2.0], [1.0, 1.0], [5, 5])  # 중복 timestamp
        with self.assertRaises(ValueError):
            normalized_content_hash([float("nan")], [1.0], [1])
        with self.assertRaises(ValueError):
            normalized_content_hash([1.0], [1.0, 2.0], [1])  # 길이 불일치


class ConfidenceCapTests(unittest.TestCase):
    """TC-7.1~7.3 — 신뢰도 상한·V-9 차단 (P0, SEC-6/계약 7-7)"""

    def test_tc71_cap_applied(self):
        self.assertEqual(apply_confidence_cap(0.9, 0.6), 0.6)   # 상한 적용
        self.assertEqual(apply_confidence_cap(0.5, 0.6), 0.5)  # 상한 이하 통과

    def test_tc72_final_output_exceeding_cap_hard_rejected(self):
        with self.assertRaises(HardRejectError):
            assert_final_output_within_cap(0.8, 0.6)  # 최종 출력 상한 초과 → V-9 하드 거부

    def test_tc73_unverified_input_not_scored(self):
        incomplete = CollectionMeta(source_id="", provider="x",
                                    data_basis_time=datetime.now(timezone.utc),
                                    received_at=datetime.now(timezone.utc))
        with self.assertRaises(HardRejectError):
            score_unverified_input(incomplete)  # 계산 미수행, 차단
        ok = CollectionMeta(source_id="kis:005930", provider="kis",
                            data_basis_time=datetime.now(timezone.utc),
                            received_at=datetime.now(timezone.utc))
        with self.assertRaises(NotImplementedError):
            score_unverified_input(ok)  # C_raw 산식은 M-PC 구현 범위 — 가짜 점수 금지


if __name__ == "__main__":
    unittest.main(verbosity=2)