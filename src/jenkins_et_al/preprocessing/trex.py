"""Convert TRex .npz tracking output into scaled CSV files.

Ported from notebooks/preprocessing/process_TRex_output.ipynb (cells
cb125626 and 08c561c8); logic is unchanged from the notebook, only moved
into a module and given a docstring/type-hint pass per the /tdd REFACTOR
checklist.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArenaScale:
    """Scale factors used to convert TRex-assumed dimensions to actual dimensions."""

    actual_length_cm: float = 14.0
    assumed_length_cm: float = 30.0
    actual_width_cm: float = 6.8
    assumed_width_cm: float = 15.0

    @property
    def x(self) -> float:
        return self.actual_length_cm / self.assumed_length_cm

    @property
    def y(self) -> float:
        return self.actual_width_cm / self.assumed_width_cm

    @property
    def average(self) -> float:
        return float(np.mean([self.x, self.y]))


#: Default arena scale used by the original notebook's saved run.
DEFAULT_SCALE = ArenaScale()

_REQUIRED_NPZ_KEYS = [
    "time",
    "frame",
    "X#wcentroid",
    "Y#wcentroid",
    "SPEED#wcentroid",
    "ACCELERATION#wcentroid",
    "missing",
    "VX",
    "VY",
    "AX",
    "AY",
    "ANGLE",
    "MIDLINE_OFFSET",
    "BORDER_DISTANCE#pcentroid",
]


def find_npz_files(input_dir: Path, search_subfolders: bool = False) -> list[Path]:
    """Return sorted TRex .npz files from `input_dir`.

    If `search_subfolders` is True, searches one level down inside child folders.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if search_subfolders:
        files = [
            file_path
            for subfolder in input_dir.iterdir()
            if subfolder.is_dir()
            for file_path in subfolder.glob("*.npz")
        ]
    else:
        files = list(input_dir.glob("*.npz"))

    return sorted(files)


def require_keys(npz_file: np.lib.npyio.NpzFile, required_keys: Iterable[str], file_path: Path) -> None:
    """Raise a helpful error if an expected TRex field is missing."""
    missing_keys = sorted(set(required_keys) - set(npz_file.files))
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise KeyError(f"{file_path.name} is missing required TRex field(s): {missing}")


def trex_npz_to_dataframe(file_path: Path, scale: ArenaScale = DEFAULT_SCALE) -> pd.DataFrame:
    """Convert one TRex .npz file to a scaled pandas DataFrame."""
    with np.load(file_path) as fish:
        require_keys(fish, _REQUIRED_NPZ_KEYS, file_path)

        return pd.DataFrame(
            {
                "time": fish["time"],
                "frame": fish["frame"],
                "x": fish["X#wcentroid"] * scale.x,
                "y": fish["Y#wcentroid"] * scale.y,
                "speed": fish["SPEED#wcentroid"] * scale.average,
                "acceleration": fish["ACCELERATION#wcentroid"] * scale.average,
                # True = fish present in frame; False = missing/lost identity
                "mask": ~fish["missing"].astype(bool),
                "derivative_x": fish["VX"] * scale.x,
                "derivative_y": fish["VY"] * scale.y,
                "acceleration_x": fish["AX"] * scale.x,
                "acceleration_y": fish["AY"] * scale.y,
                # In radians; 0 = fish horizontal, looking right
                "angle": fish["ANGLE"],
                # Use cautiously for small fish
                "tail_angle": fish["MIDLINE_OFFSET"],
                "border_distance": fish["BORDER_DISTANCE#pcentroid"] * scale.average,
            }
        )


def output_csv_path(npz_file: Path, output_dir: Path | None = None) -> Path:
    """Return the CSV path for one input .npz file.

    If `output_dir` is None, the CSV is written next to `npz_file`.
    """
    if output_dir is None:
        return npz_file.with_suffix(".csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{npz_file.stem}.csv"


def process_trex_files(
    input_dir: Path,
    scale: ArenaScale = DEFAULT_SCALE,
    output_dir: Path | None = None,
    search_subfolders: bool = False,
) -> pd.DataFrame:
    """Process all matching TRex .npz files under `input_dir` and return a summary table.

    `output_dir=None` (the default, matching the original notebook) writes each
    CSV next to its source .npz file. Pass an explicit `output_dir` to write
    elsewhere without touching files next to the raw tracking data.
    """
    npz_files = find_npz_files(input_dir, search_subfolders=search_subfolders)

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {input_dir}")

    summary_rows = []

    for npz_file in npz_files:
        fish_df = trex_npz_to_dataframe(npz_file, scale)
        csv_path = output_csv_path(npz_file, output_dir=output_dir)
        fish_df.to_csv(csv_path, index=False)

        summary_rows.append(
            {
                "input_file": str(npz_file),
                "output_file": str(csv_path),
                "rows": len(fish_df),
                "present_frames": int(fish_df["mask"].sum()),
                "missing_frames": int((~fish_df["mask"]).sum()),
            }
        )

    return pd.DataFrame(summary_rows)
