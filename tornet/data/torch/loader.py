import os
from functools import partial
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import tornet.data.preprocess as pp
from tornet.data.constants import ALL_VARIABLES
from tornet.data.loader import TornadoDataLoader, query_catalog


def numpy_to_torch(d: Dict[str, np.ndarray]):
    for key, val in d.items():
        d[key] = torch.from_numpy(np.array(val))
    return d


def _add_coordinates_torch(d, include_az, tilt_last):
    return pp.add_coordinates(
        d, include_az=include_az, tilt_last=tilt_last, backend=torch
    )


def _compute_sample_weight_transform(xy, **kwargs):
    return pp.compute_sample_weight(*xy, backend=torch, **kwargs)


def _select_keys_transform(xy, keys):
    x_selected = pp.select_keys(xy[0], keys=keys)
    return (x_selected,) + xy[1:]

def make_torch_loader(data_root: str, 
                data_type:str='train', # or 'test'
                years: list=list(range(2013,2023)),
                batch_size: int=128, 
                weights: Dict=None,
                include_az: bool=False,
                random_state:int=1234,
                select_keys: list=None,
                tilt_last: bool=True,
                from_tfds: bool=False,
                workers:int|None=None,
                pin_memory: bool=False):
    """
    Initializes torch.utils.data.DataLoader for training CNN Tornet baseline.

    data_root - location of TorNet
    data_Type - 'train' or 'test'
    years     - list of years btwn 2013 - 2022 to draw data from
    batch_size - batch size
    weights - optional sample weights, see note below
    include_az - if True, coordinates also contains az field
    random_state - random seed for shuffling files
    workers - number of workers to use for loading batches
    select_keys - Only generate a subset of keys from each tornet sample
    tilt_last - If True (default), order of dimensions is left as [batch,azimuth,range,tilt]
                If False, order is permuted to [batch,tilt,azimuth,range]
    from_tfds - Use TFDS data loader, requires this version to be
                built and TFDS_DATA_ROOT to be set.  
                See tornet/data/tdfs/tornet/README.
                If False (default), the basic loader is used

    weights is optional, if provided must be a dict of the form
      weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
    where wN,w0,w1,w2,wW are numeric weights assigned to random,
    ef0, ef1, ef2+ and warnings samples, respectively.  

    After loading TorNet samples, this does the following preprocessing:
    - Optionaly permutes order of dimensions to not have tilt last
    - Takes only last time frame
    - adds 'coordinates' variable used by CoordConv layers. If include_az is True, this
      includes r, r^{-1} (and az if include_az is True)
    - Splits sample into inputs,label
    - If weights is provided, returns inputs,label,sample_weights
    """    
    # Prefer to saturate available CPU cores by default, leaving one free for system tasks.
    if workers is None:
        cpu_total = os.cpu_count() or 1
        workers = max(cpu_total - 1, 1)

    if from_tfds:
        import tensorflow_datasets as tfds
        import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'
        ds = tfds.data_source('tornet')

        transform_list = []

        # Assumes data was saved with tilt_last=True and converts it to tilt_last=False
        if not tilt_last:
            transform_list.append(partial(pp.permute_dims, order=(0,3,1,2)))

        transform_list.extend(
            [
                pp.remove_time_dim,
                partial(_add_coordinates_torch, include_az=include_az, tilt_last=tilt_last),
                pp.split_x_y,
            ]
        )

        if weights:
            weights_kwargs = dict(weights)
            transform_list.append(partial(_compute_sample_weight_transform, **weights_kwargs))
        
        if select_keys is not None:
            transform_list.append(partial(_select_keys_transform, keys=select_keys))
            
         # Dataset, with preprocessing
        transform = transforms.Compose(transform_list)

        datasets = [TFDSTornadoDataset(ds['%s-%d' % (data_type,y)] ,transform) for y in years]
        dataset = torch.utils.data.ConcatDataset(datasets)

    else:
        file_list = query_catalog(data_root, data_type, years, random_state)

        transform_list = [
            numpy_to_torch,
            pp.remove_time_dim,
            partial(_add_coordinates_torch, include_az=include_az, tilt_last=tilt_last),
            pp.split_x_y,
        ]

        if weights:
            weights_kwargs = dict(weights)
            transform_list.append(partial(_compute_sample_weight_transform, **weights_kwargs))
        
        if select_keys is not None:
            transform_list.append(partial(_select_keys_transform, keys=select_keys))

        # Dataset, with preprocessing
        transform = transforms.Compose(transform_list)

        dataset = TornadoDataset(file_list,
                                 variables=ALL_VARIABLES,
                                 n_frames=1,
                                 tilt_last=tilt_last,
                                 transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    return loader

    
class TornadoDataset(TornadoDataLoader,Dataset):
    pass

class TFDSTornadoDataset(Dataset):
    def __init__(self,ds,transforms=None):
          self.ds=ds
          self.transforms=transforms
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = self.ds.__getitem__(idx)
        if self.transforms:
             x=self.transforms(x)
        return x
