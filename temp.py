from pathlib import Path
import numpy as np
import xarray as xr
from pathlib import Path

TARGET_VAR = "frame_labels"
OUTPUT_ROOT = Path("tornet_raw") / "retagged_shift"


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
                print(f"{TARGET_VAR} not found in {src}")
                return False
            values = ds[TARGET_VAR].values
            if values.ndim != 1:
                print(f"{TARGET_VAR} is not 1D in {src}")
                return False

            print(f"\nProcessing file: {src.name}")
            print("Before shift:", values)

            new_labels = shift_first_and_clear_rest(values)
            if new_labels is None:
                print("All labels positive, skipping export.")
                return False

            positive_mask = values > 0
            keep_indices = np.nonzero(~positive_mask)[0]
            if keep_indices.size == 0:
                print("No zeros in original labels, skipping export.")
                return False

            subset = ds.isel(time=keep_indices)
            subset[TARGET_VAR][:] = new_labels[keep_indices]

            print("After shift: ", subset[TARGET_VAR].values)

            dst.parent.mkdir(parents=True, exist_ok=True)
            subset.to_netcdf(dst)

    except Exception as exc:
        print(f"Failed to process {src}: {exc}")
        return False

    return True


def main(file_paths):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for file_path in file_paths:
        src = Path(file_path)
        if not src.exists():
            print(f"File {src} does not exist, skipping.")
            continue
        dst = OUTPUT_ROOT / src.name
        process_file(src, dst)


if __name__ == "__main__":
    # Replace with one or two files you want to test
    main([
        "tornet_raw/test/2013/TOR_130916_151034_KAMX_471045_H0.nc",
    ])
