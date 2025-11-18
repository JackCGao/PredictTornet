"""Download TorNet from Zenodo, build TFDS array_record data, and clean up."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm.auto import tqdm
import tensorflow_datasets as tfds

import tornet.data.tfds.tornet.tornet_dataset_builder  # Registers builder

ZENODO_API_BASE = "https://zenodo.org/api/records"
USER_AGENT = "PredictTornetDatasetMaker/1.0"
YEARS = tuple(range(2013, 2023))

# DOI suffixes (record IDs) pulled from the README.
ZENODO_RECORDS: List[Tuple[int, str]] = [
    (2013, "12636522"),
    (2014, "12637032"),
    (2015, "12655151"),
    (2016, "12655179"),
    (2017, "12655183"),
    (2018, "12655187"),
    (2019, "12655716"),
    (2020, "12655717"),
    (2021, "12655718"),
    (2022, "12655719"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all TorNet yearly archives from Zenodo, reformat them into "
            "TFDS array_record files, and remove the raw data afterwards."
        )
    )
    default_work_dir = os.environ.get("TORNET_ROOT") or (Path.cwd() / "tornet_raw")
    default_tfds_dir = os.environ.get("TFDS_DATA_DIR") or (Path.cwd() / "tfds_data")
    parser.add_argument(
        "--work-dir",
        default=str(default_work_dir),
        help=(
            "Directory used to unpack the TorNet tarballs. "
            "Contents are deleted on success unless --keep-raw is supplied."
        ),
    )
    parser.add_argument(
        "--tfds-data-dir",
        default=str(default_tfds_dir),
        help="Destination directory for the TFDS dataset (defaults to TFDS_DATA_DIR or ./tfds_data).",
    )
    parser.add_argument(
        "--download-cache",
        default=None,
        help="Directory used to cache the downloaded archives (defaults to <work-dir>/_downloads).",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the downloaded and extracted raw data after TFDS generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing contents in --work-dir before downloading.",
    )
    parser.add_argument(
        "--verify-checksum",
        action="store_true",
        help="Compute and verify checksums for downloads (slower but safer).",
    )
    return parser.parse_args()


def prepare_directory(path: Path, allow_delete: bool) -> None:
    if path.exists():
        if any(path.iterdir()):
            if not allow_delete:
                raise RuntimeError(
                    f"{path} is not empty. Use --force if it's safe to delete its contents."
                )
            logging.info("Clearing existing directory %s", path)
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        return
    path.mkdir(parents=True, exist_ok=True)


def fetch_record_metadata(record_id: str) -> Dict:
    url = f"{ZENODO_API_BASE}/{record_id}"
    logging.info("Fetching Zenodo metadata for record %s", record_id)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request) as response:
            return json.load(response)
    except (HTTPError, URLError) as err:
        raise RuntimeError(f"Failed to fetch metadata for record {record_id}: {err}") from err


def download_file(
    file_info: Dict, download_dir: Path, verify_checksum: bool
) -> Path:
    filename = file_info.get("key")
    if not filename:
        raise RuntimeError("Missing filename in Zenodo file metadata.")
    download_url = (
        file_info.get("links", {}).get("download")
        or file_info.get("links", {}).get("self")
    )
    if not download_url:
        raise RuntimeError(f"No download link available for {filename}.")

    destination = download_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    expected_size = file_info.get("size")
    if (
        destination.exists()
        and expected_size
        and destination.stat().st_size == expected_size
    ):
        logging.info("Skipping already-downloaded %s", destination)
        return destination

    logging.info("Downloading %s", filename)
    request = Request(download_url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request) as response, open(destination, "wb") as outfile:
            total = expected_size or int(response.headers.get("Content-Length", 0))
            progress = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=filename,
                leave=False,
            )
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                outfile.write(chunk)
                progress.update(len(chunk))
            progress.close()
    except (HTTPError, URLError) as err:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download {filename}: {err}") from err

    if verify_checksum:
        _verify_checksum(destination, file_info.get("checksum"))
    return destination


def _verify_checksum(path: Path, expected: str | None) -> None:
    if not expected:
        logging.warning("No checksum provided for %s; skipping verification.", path.name)
        return
    algo, _, expected_value = expected.partition(":")
    if not expected_value:
        logging.warning("Malformed checksum '%s' for %s; skipping.", expected, path.name)
        return

    import hashlib

    logging.info("Verifying %s with %s", path.name, algo)
    digest = hashlib.new(algo)
    with open(path, "rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_value:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {expected_value}, got {actual}"
        )


def extract_archive(archive: Path, destination: Path) -> None:
    if not tarfile.is_tarfile(archive):
        logging.warning("Skipping non-tar file %s", archive.name)
        return
    logging.info("Extracting %s", archive.name)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(str(destination.resolve())):
                raise RuntimeError(
                    f"Archive {archive.name} is trying to write outside destination."
                )
        tar.extractall(destination)


def ensure_catalog_copies(
    work_dir: Path, catalog: Path | None, created_paths: List[Path]
) -> None:
    if not catalog or not catalog.exists():
        return
    for year in YEARS:
        year_dir = work_dir / f"TorNet {year}"
        if not year_dir.exists():
            continue
        target = year_dir / "catalog.csv"
        if target.exists():
            continue
        shutil.copy2(catalog, target)
        created_paths.append(target)


def build_tfds_dataset(work_dir: Path, tfds_dir: Path) -> None:
    logging.info("Building TFDS array_record dataset inside %s", tfds_dir)
    dl_config = tfds.download.DownloadConfig(manual_dir=str(work_dir))
    tfds.data_source(
        "tornet",
        data_dir=str(tfds_dir),
        builder_kwargs={"file_format": "array_record"},
        download_and_prepare_kwargs={"download_config": dl_config},
    )


def cleanup_raw_data(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        logging.info("Removing %s", path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    work_dir = Path(args.work_dir).expanduser().resolve()
    tfds_dir = Path(args.tfds_data_dir).expanduser().resolve()
    download_cache = (
        Path(args.download_cache).expanduser().resolve()
        if args.download_cache
        else work_dir / "_downloads"
    )

    prepare_directory(work_dir, allow_delete=args.force)
    download_cache.mkdir(parents=True, exist_ok=True)
    tfds_dir.mkdir(parents=True, exist_ok=True)

    cleanup_targets: List[Path] = []
    catalog_path: Path | None = None
    build_complete = False

    try:
        for year, record_id in ZENODO_RECORDS:
            metadata = fetch_record_metadata(record_id)
            files = metadata.get("files") or []
            if not files:
                raise RuntimeError(f"No downloadable files found for year {year}.")

            for file_info in files:
                filename = file_info.get("key", "")
                destination = download_file(
                    file_info, download_cache, verify_checksum=args.verify_checksum
                )

                if filename.lower().endswith(".csv") and "catalog" in filename.lower():
                    if catalog_path is None:
                        catalog_path = work_dir / "catalog.csv"
                        shutil.copy2(destination, catalog_path)
                        cleanup_targets.append(catalog_path)
                    continue

                extract_archive(destination, work_dir)
                if download_cache not in cleanup_targets:
                    cleanup_targets.append(download_cache)

        ensure_catalog_copies(work_dir, catalog_path, cleanup_targets)
        build_tfds_dataset(work_dir, tfds_dir)
        build_complete = True
    finally:
        if build_complete and not args.keep_raw:
            cleanup_raw_data([work_dir] + [p for p in cleanup_targets if p != work_dir])


if __name__ == "__main__":
    main()
