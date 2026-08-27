"""Determinism contract for the SQL converter.

The converter must produce byte-identical output for identical input, regardless of
DuckDB thread count or scan order. Historically it did not: every entity/edge id used
``row_number() OVER ()`` with no ordering, site de-duplication used unordered ``first()``,
and wide-format relationship arrays used unordered ``list()``. Two runs on the same
export produced different files (see PR description for the recorded hashes).

The fixture is 140 real export rows (all four sources, chosen deterministically as the
lowest sample_identifiers per source), including duplicate site keys and multi-valued
arrays so that every ordering path is exercised.
"""

import hashlib
from pathlib import Path

import duckdb
import pytest

from pqg.sql_converter import convert_isamples_sql

FIXTURE = Path(__file__).parent / "fixtures" / "isamples_export_fixture.parquet"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("wide", [False, True], ids=["narrow", "wide"])
def test_two_runs_are_byte_identical(tmp_path, wide):
    outs = []
    for i in range(2):
        out = tmp_path / f"run{i}.parquet"
        convert_isamples_sql(str(FIXTURE), str(out), wide=wide, verbose=False)
        outs.append(out)
    assert _sha256(outs[0]) == _sha256(outs[1])


@pytest.mark.parametrize("wide", [False, True], ids=["narrow", "wide"])
def test_row_ids_are_unique(tmp_path, wide):
    """Regression for the narrow-format offset bug: edge ids started at agent_max and
    collided with curation / curation-agent / sample-relation entity ids."""
    out = tmp_path / "out.parquet"
    convert_isamples_sql(str(FIXTURE), str(out), wide=wide, verbose=False)
    n, d = duckdb.sql(
        f"SELECT count(*), count(DISTINCT row_id) FROM read_parquet('{out}')"
    ).fetchone()
    assert n == d, f"{n - d} duplicate row_ids"


def test_thread_count_does_not_change_output(tmp_path, monkeypatch):
    """Same bytes with 1 thread and with the default thread count."""
    import pqg.sql_converter as sc

    real_connect = duckdb.connect
    hashes = []
    for threads in (1, None):
        out = tmp_path / f"t{threads}.parquet"

        def connect(*a, _t=threads, **k):
            con = real_connect(*a, **k)
            if _t is not None:
                con.execute(f"SET threads={_t}")
            return con

        monkeypatch.setattr(sc.duckdb, "connect", connect)
        convert_isamples_sql(str(FIXTURE), str(out), wide=False, verbose=False)
        hashes.append(_sha256(out))
    assert hashes[0] == hashes[1]


def test_no_unordered_window_functions_in_source():
    """Guard against regressions: every window function must declare an ORDER BY."""
    import pqg.sql_converter as sc

    src = Path(sc.__file__).read_text()
    assert "OVER ()" not in src, "unordered window function reintroduced"
    assert "OVER (PARTITION BY e.sample_identifier)" not in src
