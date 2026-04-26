"""
Code to download or generate data
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def main(
    input_path: Path = RAW_DATA_DIR / "train.csv",
    output_path: Path = PROCESSED_DATA_DIR / "train_clean.csv",
):
    df = pd.read_csv(input_path)

    COLS_TO_KEEP = [
        'area_growth_abs_0_5h',
        'spread_bearing_cos',
        'dist_min_ci_0_5h',
        'dist_std_ci_0_5h',
        'dist_change_ci_0_5h',
        'dist_slope_ci_0_5h',
        'closing_speed_m_per_h',
        'closing_speed_abs_m_per_h',
        'projected_advance_m',
        'dist_accel_m_per_h2',
        'dist_fit_r2_0_5h',
        'alignment_cos',
        'alignment_abs',
        'cross_track_component',
        'along_track_speed',
        'event_start_hour',
        'event_start_dayofweek',
        'event_start_month',

        'time_to_hit_hours',
        'event',
    ]

    cols_present = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[cols_present]

    cyclic = {
        'event_start_hour': 24,
        'event_start_dayofweek': 7,
        'event_start_month': 12,
    }
    for col, period in cyclic.items():
        if col in df.columns:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
            df.drop(columns=[col], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    for split in ["train", "test"]:
        main(
            input_path=RAW_DATA_DIR / f"{split}.csv",
            output_path=PROCESSED_DATA_DIR / f"{split}_clean.csv",
        )
