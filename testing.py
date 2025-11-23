import os
import csv
import xarray as xr

ROOT = "tornet_raw"   # change if needed
TARGET_VAR = "frame_labels"
OUTPUT_CSV = "positive_cases_by_slice.csv"

# Store results: map file path -> list of positive slice indices
positive_cases_by_file: dict[str, list[int]] = {}

def scan_directory_for_positive_cases(root_dir: str):
    if not os.path.isdir(root_dir):
        raise SystemExit(f"Cannot find {root_dir}; update ROOT.")

    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith(".nc"):
                continue

            full_path = os.path.join(root, fname)

            try:
                ds = xr.open_dataset(full_path)
                if TARGET_VAR not in ds:
                    continue

                labels = ds[TARGET_VAR].values

                # Expect shape like (4, H, W) or (4,)
                if labels.ndim < 1 or labels.shape[0] < 4:
                    print(f"Skipping (unexpected shape) → {full_path}")
                    continue

                slices = []
                for slice_idx in range(4):
                    if (labels[slice_idx] > 0).any():
                        slices.append(slice_idx)
                if slices:
                    positive_cases_by_file[full_path] = slices

            except Exception as e:
                print(f"Error reading {full_path}: {e}")

# Run scan
scan_directory_for_positive_cases(ROOT)

# Write CSV output
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_path", "positive_slices"])
    for path, slices in sorted(positive_cases_by_file.items()):
        writer.writerow([path, " ".join(str(s) for s in slices)])

print(f"\nCSV exported → {OUTPUT_CSV}")
