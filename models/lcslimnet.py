"""Universal slimmable neural network (mipnet w/ universal slimmable layers)"""
import dataclasses
from typing import Tuple, Sequence, Optional, Union

import torch
from torch import nn
import numpy as np

from protos_compiled import model_pb2


@dataclasses.dataclass
class LCSlimNetOutputs:
    """Outputs for the MipNet.

    Attributes:
        outputs: Tensor of outputs.
    """
    outputs: torch.Tensor


class LCSlimNet(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 4, hidden_features: int = 64, use_layernorm: bool = True,
                 min_factor: float = 0.1, default_factors: Sequence[Tuple[float, float]] = None,
                 share_gradients: bool = True, latent_size: int = 64,
                 latent_codes_lods: Optional[Sequence[int]] = None,
                 init_same_latent_codes: bool = False,
                 fixed_layers: int = 0,
                 one_lc_per_lod: bool = False,
                 masking_continuous_lod: bool = False):
        """Initialize a mipnet.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            layers: Number of layers.
            hidden_features: Width of the hidden layers.
            use_layernorm: Whether to use layer norm.
            min_factor: Minimum width for the layers.
            default_factors: Default factors for the layers.
            share_gradients: Whether gradients should propagate across LoD weights.
            latent_size: Size of the latent code.
            latent_codes_lods: LoDs for the latent codes.
            init_same_latent_codes: Whether to initialize the latent codes to be the same.
            fixed_layers: Number of layers width fixed width and no latent codes.
            masking_continuous_lod: Use masking for continuous lod.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.use_layernorm = use_layernorm
        self.layers = layers
        self.fixed_layers = fixed_layers
        self.min_factor = min_factor
        self.min_nodes = int(round(self.min_factor * hidden_features))
        self.num_outputs = hidden_features - self.min_nodes + 1
        self.one_lc_per_lod = one_lc_per_lod
        self.masking_continuous_lod = masking_continuous_lod
        if layers == 1:
            input_size = in_features + latent_size
            model_layers = [nn.Linear(in_features=input_size,
                                      out_features=out_features)]
        else:
            first_layer_output = self.min_nodes if fixed_layers > 0 else hidden_features
            input_size = in_features if fixed_layers > 0 else (
                in_features + latent_size)
            model_layers = [nn.Linear(in_features=input_size,
                                      out_features=first_layer_output)]
            if use_layernorm:
                model_layers.append(nn.LayerNorm(first_layer_output))
            model_layers.append(nn.ReLU())
            for i in range(1, layers - 1):
                input_size = self.min_nodes if fixed_layers >= i else hidden_features
                output_size = self.min_nodes if fixed_layers > i else hidden_features
                if fixed_layers == i:
                    input_size += latent_size
                model_layers.append(nn.Linear(in_features=input_size,
                                              out_features=output_size))
                if use_layernorm:
                    model_layers.append(nn.LayerNorm(output_size))
                model_layers.append(nn.ReLU())
            model_layers.append(
                nn.Linear(in_features=hidden_features, out_features=out_features))
        self.model = nn.Sequential(*model_layers)
        self.default_factors = ([(0.25, 0.25), (0.5, 0.5), (0.75, 0.75),
                                (1.0, 1.0)] if default_factors is None else default_factors)
        self.factors = [(x / self.hidden_features, x / self.hidden_features)
                        for x in range(self.min_nodes, self.hidden_features + 1)]
        self.share_gradients = share_gradients
        self.init_same_latent_codes = init_same_latent_codes
        self.latent_codes_lods = [0, 128-51, 256-51, 384-51, 512 -
                                  51] if latent_codes_lods is None else latent_codes_lods
        self.one_lc_per_lod = one_lc_per_lod
        num_lods = len(self.latent_codes_lods)
        if one_lc_per_lod:
            num_lods = self.hidden_features - self.min_nodes + 1
        if init_same_latent_codes:
            latent_codes = torch.rand((num_lods, 1), dtype=torch.float32)
            latent_codes = (latent_codes /
                            torch.linalg.norm(latent_codes, dim=1, keepdim=True)).repeat(1, latent_size)
        else:
            latent_codes = torch.rand(
                (num_lods, latent_size), dtype=torch.float32)
            latent_codes = (latent_codes /
                            torch.linalg.norm(latent_codes, dim=1, keepdim=True))
        self.latent_codes = nn.Parameter(latent_codes)

    def get_latent_code(self, lod: Union[int, float]):
        """Returns a single latent code for the given lod."""
        if self.one_lc_per_lod and isinstance(lod, int):
            return self.latent_codes[lod]
        elif self.one_lc_per_lod:
            l = int(lod)
            l_next = l + 1
            i_next = min(l+1, len(self.latent_codes)-1)
            lod_frac = (lod - l) / (l_next - l)
            return (1 - lod_frac) * self.latent_codes[l] + lod_frac * self.latent_codes[i_next]
        for i in range(len(self.latent_codes_lods)-1):
            l = self.latent_codes_lods[i]
            l_next = self.latent_codes_lods[i+1]
            if isinstance(lod, int) and l == lod:
                return self.latent_codes[i]
            elif isinstance(lod, int) and l_next == lod:
                return self.latent_codes[i+1]
            elif l <= lod and lod <= l_next:
                lod_frac = (lod - l) / (l_next - l)
                return (1 - lod_frac) * self.latent_codes[i] + lod_frac * self.latent_codes[i+1]

    def _forward_linear(self, layer: nn.Linear, inputs: torch.Tensor,
                        factors: Tuple[float, float], subfactors: Tuple[float, float]) -> torch.Tensor:
        """Forward through a linear layer.

        Args:
            layer: The linear layer.
            inputs: Inputs.
            factors: Factor to subset the layer.
            subfactors: Factor for frozen weights.

        Returns:
            The output of the forward pass.
        """
        weight = layer.weight
        bias = layer.bias
        subset_shapes = (int(
            round(weight.shape[0] * factors[0])), int(round(weight.shape[1] * factors[1])))
        subset_shapes2 = (int(round(
            weight.shape[0] * subfactors[0])), int(round(weight.shape[1] * subfactors[1])))
        if self.share_gradients:
            m_weight = weight[:subset_shapes[0], :subset_shapes[1]]
            m_bias = bias[:subset_shapes[0]]
        else:
            base_weights = weight[:subset_shapes2[0], :subset_shapes2[1]]
            m_weight = torch.cat((base_weights.detach(),
                                  weight[:subset_shapes2[0], subset_shapes2[1]:subset_shapes[1]]), dim=1)
            m_weight = torch.cat((m_weight,
                                  weight[subset_shapes2[0]:subset_shapes[0], :subset_shapes[1]]), dim=0)
            base_bias = bias[:subset_shapes2[0]]
            m_bias = torch.cat((base_bias.detach(),
                                bias[subset_shapes2[0]:subset_shapes[0]]))
        return inputs @ m_weight.T + m_bias[None]

    def _forward_masking(self, inputs: torch.Tensor, fractional_lod: Union[int, float]):
        if not self.masking_continuous_lod or fractional_lod == 1:
            return inputs
        return torch.cat((inputs[:, :-1], inputs[:, -1:] * fractional_lod), dim=1)

    def _forward_layernorm(self, layer: nn.LayerNorm, inputs: torch.Tensor,
                           factor: float, subfactor: float) -> torch.Tensor:
        var, mean = torch.var_mean(inputs, dim=1, unbiased=False, keepdim=True)
        x = (inputs - mean) / torch.sqrt(var + 1e-5)
        if layer.elementwise_affine:
            weight = layer.weight
            bias = layer.bias
            subshape = int(round(weight.shape[0] * factor))
            subshape2 = int(round(weight.shape[0] * subfactor))
            if self.share_gradients:
                subweight = weight[:subshape]
                subbias = bias[:subshape]
            else:
                base_weight = weight[:subshape2]
                subweight = torch.cat(
                    (base_weight.detach(), weight[subshape2:subshape]))
                base_bias = bias[:subshape2]
                subbias = torch.cat(
                    (base_bias.detach(), bias[subshape2:subshape]))
            x = subweight[None] * x + subbias[None]
        return x

    def forward(self, inputs, lods: Optional[Sequence[Union[int, float]]] = None) -> LCSlimNetOutputs:
        if lods is None:
            lods = [int(np.round(x[0] * self.hidden_features - self.min_nodes))
                    for x in self.default_factors]
        model_outputs = []
        for lod in lods:
            lod_fraction = lod - int(lod) if isinstance(lod, float) else 1
            factor = [(lod + self.min_nodes) / self.hidden_features,
                      (lod + self.min_nodes) / self.hidden_features,
                      (lod-1 + self.min_nodes) / self.hidden_features,
                      (lod-1 + self.min_nodes) / self.hidden_features]
            lc = self.get_latent_code(lod)[None].expand(inputs.shape[0], -1)
            x = inputs
            for i in range(self.fixed_layers):
                x = self.model[3 * i](x)
                x = self.model[3 * i + 1](x)
                x = self.model[3 * i + 2](x)
            x = torch.cat([x, lc], dim=1)
            x = self._forward_linear(
                self.model[3 * self.fixed_layers], x, (factor[0], 1), (factor[2], 1))
            x = self._forward_layernorm(
                self.model[3 * self.fixed_layers + 1], x, factor[0], factor[2])
            x = self.model[3 * self.fixed_layers + 2](x)
            for i in range(self.fixed_layers + 1, self.layers - 1):
                my_factor = (factor[i % 2], factor[(i + 1) % 2])
                my_subfactor = (factor[2 + i % 2], factor[2 + (i + 1) % 2])
                x = self._forward_linear(
                    self.model[3 * i], x, my_factor, my_subfactor)
                x = self._forward_masking(x, lod_fraction)
                x = self._forward_layernorm(
                    self.model[3 * i + 1], x, my_factor[0], my_subfactor[0])
                x = self.model[3 * i + 2](x)
            x = self._forward_linear(
                self.model[-1], x, (1, factor[self.layers % 2]), (1, factor[2 + self.layers % 2]))
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return LCSlimNetOutputs(outputs=model_outputs)

    def forward_custom(self, inputs, lc_lod: Union[int, float], width_lod: Union[int, float]) -> LCSlimNetOutputs:
        model_outputs = []
        for _ in range(1):
            lod_fraction = width_lod - int(width_lod) if isinstance(width_lod, float) else 1
            factor = [(width_lod + self.min_nodes) / self.hidden_features,
                      (width_lod + self.min_nodes) / self.hidden_features,
                      (width_lod-1 + self.min_nodes) / self.hidden_features,
                      (width_lod-1 + self.min_nodes) / self.hidden_features]
            lc = self.get_latent_code(lc_lod)[None].expand(inputs.shape[0], -1)
            x = inputs
            for i in range(self.fixed_layers):
                x = self.model[3 * i](x)
                x = self.model[3 * i + 1](x)
                x = self.model[3 * i + 2](x)
            x = torch.cat([x, lc], dim=1)
            x = self._forward_linear(
                self.model[3*self.fixed_layers], x, (factor[0], 1), (factor[2], 1))
            x = self._forward_layernorm(
                self.model[3*self.fixed_layers+1], x, factor[0], factor[2])
            x = self.model[3*self.fixed_layers+2](x)
            for i in range(self.fixed_layers+1, self.layers - 1):
                my_factor = (factor[i % 2], factor[(i + 1) % 2])
                my_subfactor = (factor[2 + i % 2], factor[2 + (i + 1) % 2])
                x = self._forward_linear(
                    self.model[3 * i], x, my_factor, my_subfactor)
                x = self._forward_masking(x, lod_fraction)
                x = self._forward_layernorm(
                    self.model[3 * i + 1], x, my_factor[0], my_subfactor[0])
                x = self.model[3 * i + 2](x)
            x = self._forward_linear(
                self.model[-1], x, (1, factor[self.layers % 2]), (1, factor[2 + self.layers % 2]))
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return LCSlimNetOutputs(outputs=model_outputs)

    def _numel_linear(self, layer: nn.Linear, factors: Tuple[float, float]) -> int:
        weight = layer.weight
        bias = layer.bias
        subset_shapes = (int(
            round(weight.shape[0] * factors[0])), int(round(weight.shape[1] * factors[1])))
        subweight = weight[:subset_shapes[0], :subset_shapes[1]]
        subbias = bias[:subset_shapes[0]]
        return subweight.numel() + subbias.numel()

    def _numel_layernorm(self, layer: nn.LayerNorm, factor: float) -> int:
        if layer.elementwise_affine:
            weight = layer.weight
            bias = layer.bias
            subshape = int(round(weight.shape[0] * factor))
            subweight = weight[:subshape]
            subbias = bias[:subshape]
            return subweight.numel() + subbias.numel()
        return 0

    def lod_to_scale(self, lod: int) -> float:
        network_factor = (lod + self.min_nodes) / self.hidden_features
        scale = (2 ** (4 - 4*network_factor))
        return scale

    def scale_to_lod(self, scale: float) -> int:
        factor = (4 - np.log2(scale)) / 4
        num_nodes = int(round(factor*self.hidden_features))
        lod = num_nodes - self.min_nodes
        return lod

    def num_params(self, lod: int):
        factor = lod if isinstance(lod, Tuple) else self.factors[lod]
        total_params = 0
        for i in range(self.fixed_layers):
            total_params += sum([x.numel() for x in self.model[3 * i](x).parameters()])
            total_params += sum([x.numel() for x in self.model[3 * i + 1](x).parameters()])
        total_params += self._numel_linear(self.model[3 * self.fixed_layers], (factor[0], 1))
        total_params += self._numel_layernorm(self.model[3 * self.fixed_layers + 1], factor[0])
        for i in range(self.fixed_layers + 1, self.layers - 1):
            my_factor = (factor[i % 2], factor[(i + 1) % 2])
            total_params += self._numel_linear(self.model[3 * i], my_factor)
            total_params += self._numel_layernorm(
                self.model[3 * i + 1], my_factor[0])
        total_params += self._numel_linear(
            self.model[-1], (1, factor[self.layers % 2]))
        return total_params

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
