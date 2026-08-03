"""Unit tests for jenkins_et_al.preprocessing.headfixed_bhv.

Ported from notebooks/preprocessing/process_headfixed_BHV.ipynb without a
golden-output regression test: the user opted to skip capturing/comparing
against the real data drive for this port. These tests exercise the ported
logic against small synthetic inputs instead of verifying against a captured
run of the original notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.preprocessing import headfixed_bhv


def test_running_std_matches_pandas_rolling_std():
    trace = np.array([1.0, 2.0, 3.0, 10.0, 2.0, 1.0, 1.0, 1.0])
    result = headfixed_bhv.running_std(trace, window_size_samples=3)
    expected = pd.Series(trace).rolling(window=3).std().fillna(0).to_numpy()
    assert np.allclose(result, expected)


def test_running_std_fills_warmup_nans_with_zero():
    trace = np.array([5.0, 5.0, 5.0])
    result = headfixed_bhv.running_std(trace, window_size_samples=5)
    assert np.array_equal(result, np.zeros(3))


def test_find_bouts_returns_empty_for_constant_trace():
    angles = np.zeros(200)
    assert headfixed_bhv.find_bouts(angles, fs=60.85) == []


def test_find_bouts_detects_a_single_movement_bout():
    fs = 60.0
    angles = np.zeros(300)
    angles[100:110] = np.linspace(0, 5, 10)  # a clear movement segment
    bouts = headfixed_bhv.find_bouts(angles, fs=fs, min_length=0.05)
    assert len(bouts) == 1
    bout = bouts[0]
    # the running-std (window=50) vigour trace stays elevated for a while
    # after the movement itself ends, so the detected bout starts within the
    # movement segment and extends well past it -- it should not, however,
    # extend into the untouched trailing baseline.
    assert 99 <= bout[0] <= 109
    assert bout[-1] < 250


def test_get_stimulus_name_strips_movie_prefix_and_angles_suffix():
    file = "/data/7dpf_fb/230713/movie_2023-05-26_2_cad_2.5_2_angles.npy"
    assert headfixed_bhv.get_stimulus_name(file) == "2_cad_2.5_2"


@pytest.mark.parametrize(
    "stim, expected",
    [
        ("cad_2.5", "cad_2.5mm"),
        ("cad_25", "cad_25um"),
        ("kw", "kw"),
    ],
)
def test_fix_stim_name(stim, expected):
    assert headfixed_bhv.fix_stim_name(stim) == expected


def _write_angles_file(path, values):
    np.save(path, np.asarray(values, dtype=float))


def test_build_bhv_dataframe_pads_short_traces_and_sets_columns(tmp_path):
    fish_dir = tmp_path / "230713"
    fish_dir.mkdir()
    _write_angles_file(
        fish_dir / "movie_2023-05-26_2_cad_2.5_2_angles.npy",
        np.arange(10, dtype=float),
    )

    df = headfixed_bhv.build_bhv_dataframe(tmp_path, sampling_frequency=60.85, trace_length=6000)

    assert list(df.columns) == ["angles", "vigour", "bouts", "fish_id", "stimulus", "trial_number"]
    assert len(df) == 1
    row = df.iloc[0]
    assert len(row["angles"]) == 6000
    assert row["fish_id"] == "230713"
    assert row["stimulus"] == "2_cad_2.5mm"
    assert row["trial_number"] == "2"


def test_build_bhv_dataframe_normalises_to_zero_mean(tmp_path):
    fish_dir = tmp_path / "230713"
    fish_dir.mkdir()
    _write_angles_file(
        fish_dir / "movie_2023-05-26_2_cad_2.5_2_angles.npy",
        np.full(6000, 7.0),
    )

    df = headfixed_bhv.build_bhv_dataframe(tmp_path, trace_length=6000)
    angles = np.array(df.iloc[0]["angles"])
    assert np.allclose(angles, 0.0)


def test_build_bhv_dataframe_combines_multiple_fish(tmp_path):
    for fish_id, stim_token in [("230713", "2_cad_2.5_1"), ("230714", "2_kw_1")]:
        fish_dir = tmp_path / fish_id
        fish_dir.mkdir()
        _write_angles_file(
            fish_dir / f"movie_2023-05-26_{stim_token}_angles.npy",
            np.zeros(100),
        )

    df = headfixed_bhv.build_bhv_dataframe(tmp_path)
    assert set(df["fish_id"]) == {"230713", "230714"}
    assert len(df) == 2


def test_output_csv_path_is_sibling_with_suffix(tmp_path):
    folders_path = tmp_path / "7dpf_fb"
    expected = tmp_path / "7dpf_fb_reorientation_df.csv"
    assert headfixed_bhv.output_csv_path(folders_path) == expected


def test_process_headfixed_bhv_writes_csv_to_default_path(tmp_path):
    folders_path = tmp_path / "7dpf_fb"
    fish_dir = folders_path / "230713"
    fish_dir.mkdir(parents=True)
    _write_angles_file(
        fish_dir / "movie_2023-05-26_2_cad_2.5_2_angles.npy",
        np.zeros(100),
    )

    headfixed_bhv.process_headfixed_bhv(folders_path)

    output_path = tmp_path / "7dpf_fb_reorientation_df.csv"
    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert saved.iloc[0]["fish_id"] == 230713


def test_process_headfixed_bhv_writes_csv_to_custom_output_path(tmp_path):
    folders_path = tmp_path / "7dpf_fb"
    fish_dir = folders_path / "230713"
    fish_dir.mkdir(parents=True)
    _write_angles_file(
        fish_dir / "movie_2023-05-26_2_cad_2.5_2_angles.npy",
        np.zeros(100),
    )
    output_path = tmp_path / "custom_output.csv"

    headfixed_bhv.process_headfixed_bhv(folders_path, output_path=output_path)

    assert output_path.exists()
