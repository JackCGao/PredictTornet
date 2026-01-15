from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from tornet.models.torch.cnn_baseline import TornadoLikelihood


class TornadoSeqClassifier(nn.Module):
    """
    Sequence model that reuses the baseline TornadoLikelihood CNN on multiple time slices,
    pools each slice, then classifies the pooled sequence with an LSTM.

    Expected input format:
      - data dict contains radar variables with shape [batch, time, tilt, az, rng]
        (tilt_last=False) or [batch, time, az, rng, tilt] (tilt_last=True).
      - 'coordinates' should have a matching time dimension: [batch, time, coord, az, rng]
        if tilt_last=False, or [batch, time, az, rng, coord] if tilt_last=True.
      - If a sequence_length is set, only the most recent sequence_length slices are used.

    The CNN output for each time slice is globally max pooled, producing a scalar per slice,
    which is then fed to an LSTM for binary classification (logits with two channels).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        c_shape: Tuple[int, ...],
        input_variables: List[str],
        include_range_folded: bool = True,
        start_filters: int = 48,
        kernel_size: int = 3,
        n_blocks: int = 4,
        convs_per_block: Iterable[int] | None = None,
        drop_rate: float = 0.1,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        sequence_length: int | None = None,
        tilt_last: bool = False,
    ):
        super().__init__()
        self.input_variables = input_variables
        self.include_range_folded = include_range_folded
        self.sequence_length = sequence_length
        self.tilt_last = tilt_last

        self.cnn = TornadoLikelihood(
            shape=shape,
            c_shape=c_shape,
            input_variables=input_variables,
            include_range_folded=include_range_folded,
            start_filters=start_filters,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            convs_per_block=convs_per_block,
            drop_rate=drop_rate,
        )
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(lstm_hidden_size, 2)

    def _select_time_indices(self, total_time: int, device: torch.device) -> torch.Tensor:
        if self.sequence_length is None or self.sequence_length >= total_time:
            return torch.arange(total_time, device=device)
        start = total_time - self.sequence_length
        return torch.arange(start, total_time, device=device)

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        # Ensure all required keys exist
        for v in self.input_variables:
            if v not in data:
                raise KeyError(f"Input missing variable '{v}' required by the model.")
        if "coordinates" not in data:
            raise KeyError("Input missing 'coordinates' required by the model.")

        sample_var = data[self.input_variables[0]]
        if sample_var.dim() < 5:
            raise ValueError("Expected time dimension present; got tensor with shape %s" % (sample_var.shape,))
        bsz, time_len = sample_var.shape[0], sample_var.shape[1]
        device = sample_var.device
        time_idx = torch.arange(time_len, device=device)

        pooled_slices: List[torch.Tensor] = []

        for t in time_idx:
            slice_dict: Dict[str, torch.Tensor] = {}
            for v in self.input_variables:
                x = data[v][:, t]  # shape depends on tilt_last flag
                if self.tilt_last and x.dim() == 4:
                    # [batch, az, rng, tilt] -> [batch, tilt, az, rng]
                    x = x.permute(0, 3, 1, 2)
                slice_dict[v] = x

            coords = data["coordinates"][:, t]
            if self.tilt_last and coords.dim() == 4:
                # [batch, az, rng, coord] -> [batch, coord, az, rng]
                coords = coords.permute(0, 3, 1, 2)
            slice_dict["coordinates"] = coords

            # Pass through baseline CNN
            heatmap = self.cnn(slice_dict)  # [batch, 1, H, W]
            pooled = F.max_pool2d(heatmap, kernel_size=heatmap.size()[2:])  # [batch, 1, 1, 1]
            pooled = pooled.view(bsz, 1)  # [batch, 1 feature]
            pooled_slices.append(pooled)

        # Stack sequence: [batch, seq_len, features]
        seq = torch.stack(pooled_slices, dim=1)
        lstm_out, _ = self.lstm(seq)
        # Use last time step hidden state
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits
