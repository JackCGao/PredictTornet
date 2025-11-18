"""tornet dataset."""
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

import sys
from tornet.data.loader import read_file

class Builder(tfds.core.GeneratorBasedBuilder):
  """
  DatasetBuilder for tornet.  See README.md in this directory for how to build
  """

  VERSION = tfds.core.Version('1.1.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.1.0': 'Label Fix, added start/end times',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Find instructions to download TorNet on https://github.com/mit-ll/tornet
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'DBZ': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'VEL': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'KDP': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'RHOHV': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'ZDR': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'WIDTH': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'range_folded_mask': tfds.features.Tensor(shape=(4, 120, 240, 2),dtype=np.float32,encoding='zlib'),
            'label': tfds.features.Tensor(shape=(4,),dtype=np.uint8),
            'category': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'event_id': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'ef_number': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'az_lower': tfds.features.Tensor(shape=(1,),dtype=np.float32),
            'az_upper': tfds.features.Tensor(shape=(1,),dtype=np.float32),
            'rng_lower': tfds.features.Tensor(shape=(1,),dtype=np.float32),
            'rng_upper': tfds.features.Tensor(shape=(1,),dtype=np.float32),
            'time': tfds.features.Tensor(shape=(4,),dtype=np.int64),
            'tornado_start_time': tfds.features.Tensor(shape=(1,),dtype=np.int64),
            'tornado_end_time': tfds.features.Tensor(shape=(1,),dtype=np.int64),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://github.com/mit-ll/tornet',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    
    # Assumes data is already downloaded and extracted from tar files
    # manual_dir should point to where tar files were extracted
    archive_path = pathlib.Path(dl_manager.manual_dir)

    split_dirs = self._resolve_split_directories(archive_path)
    if not split_dirs:
      raise FileNotFoundError(
          f'Could not find TorNet train/test directories under {archive_path}. '
          'See README.md for the expected layout.')

    return {
        split_name: self._generate_examples(split_path)
        for split_name, split_path in split_dirs
    }

  def _resolve_split_directories(self, archive_path: pathlib.Path):
    """Return a list of (split-name, path) tuples for the available data."""
    split_dirs = []

    def _add_split(split, year, path):
      split_dirs.append((f'{split}-{year}', path))

    # Layout 1: <root>/train/<year>, <root>/test/<year>
    has_standard_layout = any(
        (archive_path / split).exists() for split in ('train', 'test'))
    if has_standard_layout:
      for split in ('train', 'test'):
        base = archive_path / split
        if not base.exists():
          continue
        for year_dir in sorted(base.iterdir()):
          if not year_dir.is_dir():
            continue
          try:
            year = int(year_dir.name)
          except ValueError:
            continue
          _add_split(split, year, year_dir)
      return split_dirs

    # Layout 2: <root>/TorNet <year>/<split>/<year>
    for year_root in sorted(archive_path.glob('TorNet *')):
      if not year_root.is_dir():
        continue
      parts = year_root.name.split()
      try:
        year = int(parts[-1])
      except (ValueError, IndexError):
        continue
      for split in ('train', 'test'):
        candidate = year_root / split / str(year)
        if candidate.exists():
          _add_split(split, year, candidate)

    return split_dirs

  def _generate_examples(self, path):
    """Yields examples."""
    # Yields (key, example) tuples from the dataset
    # key is the original netcdf filename
    data_type = path.parent.name # 'train' or 'test'
    year = int(os.path.basename(path)) # year
    catalog_path = path / '../../catalog.csv'
    catalog = pd.read_csv(catalog_path,parse_dates=['start_time','end_time'])
    catalog = catalog[catalog['type']==data_type]
    catalog = catalog[catalog.end_time.dt.year.isin([year])]
    catalog = catalog.sample(frac=1,random_state=1234) # shuffle
    #catalog = catalog.iloc[:10] # testing

    for f in catalog.filename:
      # files are relative to dl_manager.manual_dir
      yield f, read_file(path / ('../../'+f),n_frames=4)
