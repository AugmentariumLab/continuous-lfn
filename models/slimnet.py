"""Universal slimmable neural network (mipnet w/ universal slimmable layers)"""
import dataclasses
from typing import Tuple, Sequence, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from protos_compiled import model_pb2


@dataclasses.dataclass
class SlimNetOutputs:
    """Outputs for the slimmable network.

    Attributes:
        outputs: Tensor of outputs.
    """
    outputs: torch.Tensor


class SlimNet(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 4, hidden_features: int = 64, use_layernorm: bool = True,
                 min_factor: float = 0.1, default_factors: Sequence[Tuple[float, float]] = None,
                 share_gradients: bool = False, fixed_layers: int = 0,
                 masking_continuous_lod: bool = False):
        """Initialize a slimmable network.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            layers: Number of total layers.
            hidden_features: Maximum width of hidden layers
            use_layernorm: Whether to use layer norm.
            factors: Width factor for each layer.
            share_gradients: Whether gradients should propagate across LoD weights.
            fixed_layers: Number of initial layers with fixed widths;
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
        self.masking_continuous_lod = masking_continuous_lod
        if layers == 1:
            model_layers = [nn.Linear(in_features=in_features,
                                      out_features=out_features)]
        else:
            output_size = self.min_nodes if fixed_layers > 0 else hidden_features
            model_layers = [nn.Linear(in_features=in_features,
                                      out_features=output_size)]
            if use_layernorm:
                model_layers.append(nn.LayerNorm(output_size))
            model_layers.append(nn.ReLU())
            for i in range(1, layers - 1):
                input_size = self.min_nodes if fixed_layers >= i else hidden_features
                output_size = self.min_nodes if fixed_layers > i else hidden_features
                model_layers.append(nn.Linear(in_features=input_size,
                                              out_features=output_size))
                if use_layernorm:
                    model_layers.append(nn.LayerNorm(output_size))
                model_layers.append(nn.ReLU())
            model_layers.append(
                nn.Linear(in_features=hidden_features, out_features=out_features))
        self.model = nn.Sequential(*model_layers)
        self.num_outputs = hidden_features - self.min_nodes + 1
        self.default_factors = ([(0.25, 0.25), (0.5, 0.5), (0.75, 0.75),
                                (1.0, 1.0)] if default_factors is None else default_factors)
        self.factors = [(x / self.hidden_features, x / self.hidden_features)
                        for x in range(self.min_nodes, self.hidden_features + 1)]
        self.share_gradients = share_gradients
        self.neuron_mask_size = 1
        self.neuron_mask_size_warned = False

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
            return F.linear(inputs, m_weight, m_bias)
            # Testing pad to a multiple of 8:
            # ss0_round_up = 8 * ((subset_shapes[0] + 7) // 8)
            # ss1_round_up = 8 * ((subset_shapes[1] + 7) // 8)
            # inputs_padded = F.pad(inputs, (0, ss1_round_up - inputs.shape[1]))
            # m_weight = F.pad(weight[:subset_shapes[0], :subset_shapes[1]],
            #                  (0, ss1_round_up - subset_shapes[1], 0, ss0_round_up - subset_shapes[0]))
            # m_bias = F.pad(bias[:subset_shapes[0]], (0, ss0_round_up - subset_shapes[0]))
            # return F.linear(inputs_padded, m_weight, m_bias)[:, :subset_shapes[0]]
        base_weights = weight[:subset_shapes2[0], :subset_shapes2[1]]
        m_weight = torch.cat((base_weights.detach(),
                              weight[:subset_shapes2[0], subset_shapes2[1]:subset_shapes[1]]), dim=1)
        m_weight = torch.cat((m_weight,
                              weight[subset_shapes2[0]:subset_shapes[0], :subset_shapes[1]]), dim=0)
        base_bias = bias[:subset_shapes2[0]]
        m_bias = torch.cat((base_bias.detach(),
                            bias[subset_shapes2[0]:subset_shapes[0]]))
        return F.linear(inputs, m_weight, m_bias)

    def _forward_masking(self, inputs: torch.Tensor, fractional_lod: Union[int, float]):
        if not self.masking_continuous_lod or fractional_lod == 1:
            return inputs
        if self.neuron_mask_size > 1 and not self.neuron_mask_size_warned:
            print(f"Warning: masking {self.neuron_mask_size} neurons at a time.")
            self.neuron_mask_size_warned = True
        return torch.cat((inputs[:, :-self.neuron_mask_size],
                          inputs[:, -self.neuron_mask_size:] * fractional_lod), dim=1)

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

    def forward(self, inputs, lods: Optional[Sequence[Union[int, float]]] = None) -> SlimNetOutputs:
        if lods is None:
            lods = [int(np.round(x[0] * self.hidden_features - self.min_nodes))
                    for x in self.default_factors]
        model_outputs = []
        for lod in lods:
            lod_ceil = int(np.ceil(lod))
            lod_fraction = lod - \
                (lod_ceil - 1) if isinstance(lod, float) else 1
            factor = [(lod_ceil + self.min_nodes) / self.hidden_features,
                      (lod_ceil + self.min_nodes) / self.hidden_features,
                      (lod_ceil-1 + self.min_nodes) / self.hidden_features,
                      (lod_ceil-1 + self.min_nodes) / self.hidden_features]
            x = inputs
            for i in range(self.fixed_layers):
                x = self.model[3 * i](x)
                x = self.model[3 * i + 1](x)
                x = self.model[3 * i + 2](x)
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
        return SlimNetOutputs(outputs=model_outputs)

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
            total_params += sum([x.numel()
                                for x in self.model[3 * i].parameters()])
            total_params += sum([x.numel()
                                for x in self.model[3 * i + 1].parameters()])
        total_params += self._numel_linear(
            self.model[3 * self.fixed_layers], (factor[0], 1))
        total_params += self._numel_layernorm(
            self.model[3 * self.fixed_layers + 1], factor[0])
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
