from pathlib import Path

import numpy as np
import xarray as xr

TARGET_VAR = "frame_labels"
INPUT_ROOT = Path("tornet_raw")
OUTPUT_ROOT = INPUT_ROOT / "retagged_shift"


def shift_first_and_clear_rest(labels):
    labels = labels.copy()

    pos = np.where(labels == 1)[0]
    if len(pos) == 0:
        return labels
    if len(pos) == len(labels):
        return None  # Special flag: all slices positive, skip

    sequences = []
    start = pos[0]

    # find sequences of consecutive 1's
    for i in range(1, len(pos)):
        if pos[i] != pos[i - 1] + 1:
            sequences.append((start, pos[i - 1]))
            start = pos[i]
    sequences.append((start, pos[-1]))

    # process each sequence
    for (s, e) in sequences:
        # clear the entire sequence
        for i in range(s, e + 1):
            labels[i] = 0

        # If the first one was at index 0:
        # → do NOT shift; leave all zeros
        if s == 0:
            continue

        # Otherwise shift the first 1 backward
        labels[s - 1] = 1

    return labels


def process_file(src: Path, dst: Path) -> bool:
    try:
        with xr.open_dataset(src) as ds:
            if TARGET_VAR not in ds:
                return False
            values = ds[TARGET_VAR].values
            if values.ndim != 1:
                return False
            new_labels = shift_first_and_clear_rest(values)
            if new_labels is None:
                return False  # skip export for all-positive samples
            positive_mask = values > 0
            keep_indices = np.nonzero(~positive_mask)[0]
            if keep_indices.size == 0:
                return False
            subset = ds.isel(time=keep_indices)
            subset[TARGET_VAR][:] = new_labels[keep_indices]
            dst.parent.mkdir(parents=True, exist_ok=True)
            subset.to_netcdf(dst)
    except Exception as exc:
        print(f"Failed to process {src}: {exc}")
        return False
    return True


def main():
    if not INPUT_ROOT.exists():
        raise SystemExit(f"Input root {INPUT_ROOT} does not exist.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    files_processed = 0
    for src in INPUT_ROOT.rglob("*.nc"):
        rel = src.relative_to(INPUT_ROOT)
        dst = OUTPUT_ROOT / rel
        if process_file(src, dst):
            files_processed += 1
            print(f"Retagged {rel}")

    if files_processed == 0:
        raise SystemExit("No NetCDF files were retagged. Check the dataset path or contents.")
    print(f"Finished retagging {files_processed} files.")


if __name__ == "__main__":
    main()
