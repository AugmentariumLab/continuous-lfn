"""This is a network which uses latent codes for each LOD."""
import dataclasses
from typing import Tuple, Sequence, Optional, Union
from numpy import isin

import torch
from torch import nn

from protos_compiled import model_pb2

@dataclasses.dataclass
class LCNetOutputs:
    """Outputs for the LCNet.

    Attributes:
        outputs: Tensor of outputs as (B, LOD, out_features).
    """
    outputs: torch.Tensor

class LCNet(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 4, hidden_features: int = 64, use_layernorm: bool = True,
                 latent_size: int = 64, num_lods: int = 4):
        """Initialize a mipnet.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            layers: Number of layers
            hidden_features: Features in hidden layers.
            use_layernorm: Whether to use layer norm.
            latent_size: Size of latent code.
            num_lods: Number of LODs.
        """
        super().__init__()
        if layers == 1:
            model_layers = [nn.Linear(in_features=in_features + latent_size,
                                      out_features=out_features)]
        else:
            model_layers = [nn.Linear(in_features=in_features + latent_size,
                                      out_features=hidden_features)]
            if use_layernorm:
                model_layers.append(nn.LayerNorm(hidden_features))
            model_layers.append(nn.ReLU())
            for _ in range(layers - 2):
                model_layers.append(nn.Linear(in_features=hidden_features,
                                              out_features=hidden_features))
                if use_layernorm:
                    model_layers.append(nn.LayerNorm(hidden_features))
                model_layers.append(nn.ReLU())
            model_layers.append(
                nn.Linear(in_features=hidden_features, out_features=out_features))
        self.model = nn.Sequential(*model_layers)
        latent_codes = torch.rand((num_lods, latent_size), dtype=torch.float32)
        latent_codes = latent_codes / torch.linalg.norm(latent_codes, dim=1, keepdim=True)
        self.latent_codes = nn.Parameter(latent_codes)        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.use_layernorm = use_layernorm
        self.layers = layers
        self.latent_size = latent_size
        self.num_outputs = num_lods

    def get_latent_code(self, lod: Union[int, float]):
        """Returns a single latent code for the given lod."""
        if isinstance(lod, int):
            return self.latent_codes[lod]
        elif isinstance(lod, float):
            lod_floor = max(int(lod), 0)
            lod_ceil = min(lod_floor + 1, self.latent_codes.shape[0]-1)
            lod_frac = lod - lod_floor
            return (1 - lod_frac) * self.latent_codes[lod_floor] + lod_frac * self.latent_codes[lod_ceil]
        else:
            raise ValueError(f"Invalid type for lod: {type(lod)}")

    def forward(self, inputs, lods: Optional[Sequence[Union[int, float]]] = None) -> LCNetOutputs:
        if lods is None:
            lods = range(self.num_outputs)
        outputs = []
        for lod in lods:
            lc = self.get_latent_code(lod)[None].expand(inputs.shape[0], -1)
            x = torch.cat([inputs, lc], dim=1)
            output = self.model(x)
            outputs.append(output)
        return LCNetOutputs(torch.stack(outputs, dim=1))

    def num_params(self, lod: int):
        return 0

    def get_proto(self) -> model_pb2.Model.Network:
        """Get proto containing weights.

        Returns:
            The proto.
        """
        network = model_pb2.Model.Network()
        state_dict = self.state_dict()
        for k, v in state_dict.items():
            v_np = v.cpu().numpy()
            network.state_dict[k].shape.extend(v_np.shape)
            network.state_dict[k].values.extend(v_np.ravel())
        exit(0)
