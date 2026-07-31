"""Regression test for jenkins_et_al.preprocessing.trex against a golden capture
of notebooks/preprocessing/process_TRex_output.ipynb (commit 2d19f3a).

The golden capture (golden/process_TRex_output/) was produced by executing the
notebook's own cell source directly, redirected to write CSVs under golden/
instead of next to the source .npz files (the notebook's own default output
location, which holds real production data this test must not touch).

Skipped entirely if the source data drive isn't mounted, since the underlying
.npz files are not part of this repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.preprocessing import trex

INPUT_DIR = Path("/Volumes/LaCie/free_swimming/7dpf_TRex/adenosine_2.5mM/control")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "process_TRex_output"

pytestmark = pytest.mark.skipif(
    not INPUT_DIR.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def ported_summary(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("trex_csvs")
    summary = trex.process_trex_files(
        input_dir=INPUT_DIR,
        scale=trex.DEFAULT_SCALE,
        output_dir=output_dir,
        search_subfolders=True,
    )
    return summary, output_dir


def test_arena_scale_matches_notebook_defaults():
    scale = trex.ArenaScale()
    assert scale.x == pytest.approx(14.0 / 30.0)
    assert scale.y == pytest.approx(6.8 / 15.0)
    assert scale.average == pytest.approx(np.mean([14.0 / 30.0, 6.8 / 15.0]))


def test_summary_matches_golden_file_count_and_frame_counts(ported_summary):
    summary, _ = ported_summary
    golden_summary = pd.read_csv(GOLDEN_DIR / "summary.csv")

    summary_by_stem = summary.assign(stem=summary["input_file"].map(lambda p: Path(p).stem)).set_index("stem")
    golden_by_stem = golden_summary.assign(stem=golden_summary["input_file"].map(lambda p: Path(p).stem)).set_index("stem")

    assert set(summary_by_stem.index) == set(golden_by_stem.index)
    aligned_golden = golden_by_stem.loc[summary_by_stem.index]

    assert (summary_by_stem["rows"].values == aligned_golden["rows"].values).all()
    assert (summary_by_stem["present_frames"].values == aligned_golden["present_frames"].values).all()
    assert (summary_by_stem["missing_frames"].values == aligned_golden["missing_frames"].values).all()


def test_every_output_csv_matches_golden_csv(ported_summary):
    _, output_dir = ported_summary
    golden_csv_dir = GOLDEN_DIR / "csvs"

    golden_csvs = sorted(golden_csv_dir.glob("*.csv"))
    assert len(golden_csvs) == 166, "expected file count changed -- check INPUT_DIR contents"

    for golden_csv in golden_csvs:
        ported_csv = output_dir / golden_csv.name
        assert ported_csv.exists(), f"port did not produce {golden_csv.name}"

        golden_df = pd.read_csv(golden_csv)
        ported_df = pd.read_csv(ported_csv)

        assert list(golden_df.columns) == list(ported_df.columns)

        numeric_cols = golden_df.select_dtypes(include=[np.number]).columns
        assert np.allclose(
            golden_df[numeric_cols].values,
            ported_df[numeric_cols].values,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ), f"numeric mismatch in {golden_csv.name}"

        assert (golden_df["mask"] == ported_df["mask"]).all(), f"mask mismatch in {golden_csv.name}"
