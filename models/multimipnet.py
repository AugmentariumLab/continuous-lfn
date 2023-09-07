"""Wrapper model which consists of multiple submodels."""
import dataclasses
from typing import Tuple, Optional, Sequence

import torch
from torch import nn
import numpy as np

from models import mlp
from utils import my_torch_utils
from protos_compiled import model_pb2


@dataclasses.dataclass
class MultiMipnetOutputs:
    """Outputs for the multi model MLP.

    Attributes:
        model_outputs: The final outputs.
        selection_indices: Indices for which model to use.
        selection_logits: Outputs for the selection model.
        selection_probabilities: Probabilities for which model to use.
    """
    model_outputs: torch.Tensor
    selection_indices: torch.Tensor
    selection_logits: torch.Tensor
    selection_probabilities: torch.Tensor


class MultiMipnetLinear(nn.Module):
    def __init__(self, num_models: int, in_features: int, out_features: int):
        super().__init__()
        sqrt_k = np.sqrt(1 / num_models)

        self.weights = nn.Parameter(
            2 * sqrt_k * torch.rand((num_models, out_features, in_features), dtype=torch.float32) - sqrt_k)
        self.biases = nn.Parameter(
            2 * sqrt_k * torch.rand((num_models, out_features), dtype=torch.float32) - sqrt_k)

    def forward(self, inputs: torch.Tensor, indices: torch.Tensor, factors: Tuple[float, float]):
        weight = self.weights
        bias = self.biases
        subset_shapes = (int(
            round(weight.shape[1] * factors[0])), int(round(weight.shape[2] * factors[1])))
        subweight = weight[indices, :subset_shapes[0], :subset_shapes[1]]
        subbias = bias[indices, :subset_shapes[0]]
        return torch.matmul(subweight, inputs[:, :, None])[:, :, 0] + subbias

    def num_params(self, factors: Tuple[float, float]):
        weight = self.weights
        bias = self.biases
        subset_shapes = (int(
            round(weight.shape[1] * factors[0])), int(round(weight.shape[2] * factors[1])))
        subweight_shape = (weight.shape[0] *
                           subset_shapes[0] * subset_shapes[1])
        subbias_shape = bias.shape[0] * subset_shapes[0]
        return subweight_shape + subbias_shape


class MultiMipnet(nn.Module):
    def __init__(self, num_models: int = 64, in_features: int = 6, out_features: int = 3,
                 layers: int = 5, hidden_features: int = 64,
                 selection_layers: int = 5, selection_hidden_features: int = 64,
                 lerp_value: float = 0.5, selection_mode: str = "cylinder",
                 factors: Optional[Sequence[Tuple[float, float]]] = None):
        """Initialize a multi mlp.

        Args:
            num_models: Number of submodels.
            in_features: Input features.
            out_features: Output features.
            layers: Layers per submodel.
            hidden_features: Features per submodel.
            selection_layers: Number of layers for selection network.
            selection_hidden_features: Number of hidden features for selection network.
            lerp_value: Value for lerp.
            selection_mode: Mode for selection.
        """
        super().__init__()
        if num_models <= 0:
            raise ValueError(f"Num models is <= 0: {num_models}")
        self.num_models = num_models
        self.in_features = in_features
        self.out_features = out_features
        self.layers = layers
        self.hidden_features = hidden_features
        self.selection_layers = selection_layers
        self.selection_hidden_features = selection_hidden_features
        self.selection_mode = selection_mode
        self.lerp_value = lerp_value
        self.num_angle_sections = 8
        self.num_height_sections = 8
        self.max_height = 1.0
        self.min_height = -3.0
        self.radius = 1.0
        self.num_outputs = 3
        self.selection_model = mlp.MLP(in_features=in_features, out_features=num_models, layers=selection_layers,
                                       hidden_features=selection_hidden_features) if selection_mode == "mlp" else None
        model_layers = [MultiMipnetLinear(
            num_models=num_models,
            in_features=in_features,
            out_features=hidden_features
        ), nn.ReLU()]
        for _ in range(layers - 2):
            model_layers.append(MultiMipnetLinear(
                num_models=num_models,
                in_features=hidden_features,
                out_features=hidden_features
            ))
            model_layers.append(nn.ReLU())
        model_layers.append(MultiMipnetLinear(
            num_models=num_models,
            in_features=hidden_features,
            out_features=out_features
        ))
        self.model_layers = nn.ModuleList(model_layers)

        self.factors = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75),
                        (1.0, 1.0)] if factors is None else factors
        self.num_outputs = len(self.factors)

    def freeze_selection_model(self):
        if self.selection_model is not None:
            for parameter in self.selection_model.parameters():
                parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor, factors=None) -> MultiMipnetOutputs:
        """Forward pass.

        Args:
            inputs: Inputs as (B, F) tensor.

        Returns:
            The model outputs as (B, F) tensor.
        """
        if factors is None:
            factors = [self.factors[0] + (0, 0)] + [x + y for x,
                                                    y in zip(self.factors[1:], self.factors[:-1])]
        if self.selection_mode == "angle":
            angles = torch.atan2(inputs[:, 2], inputs[:, 0])
            angles = torch.fmod(angles + 2 * np.pi, 2 *
                                np.pi) / (2 * np.pi) * self.num_models
            angles = torch.floor(angles)
            selection_indices = angles.long()
            selection_logits = torch.ones(
                (angles.shape[0], self.num_models), dtype=inputs.dtype, device=inputs.device)
            selection_probabilities = torch.softmax(selection_logits, dim=1)
        elif self.selection_mode == "mlp":
            selection_logits = self.selection_model(inputs)
            selection_probabilities = torch.softmax(selection_logits, dim=1)
            selection_indices = torch.argmax(selection_probabilities, dim=1)
        elif self.selection_mode == "cylinder" or self.selection_mode == "cylinder_out":
            num_angle_sections = self.num_angle_sections
            num_height_sections = self.num_height_sections
            max_height = self.max_height
            min_height = self.min_height
            radius = self.radius
            assert num_angle_sections * num_height_sections >= self.num_models, "Invalid"
            ray_direction = inputs[:, :3]
            ray_points = torch.cross(inputs[:, 3:6], inputs[:, :3])
            t = my_torch_utils.solve_quadratic(
                ray_direction[:, [0, 2]].pow(2).sum(1),
                (2 * ray_points[:, [0, 2]] * ray_direction[:, [0, 2]]).sum(-1),
                ray_points[:, [0, 2]].pow(2).sum(1) - radius * radius,
                use_minus=(self.selection_mode == "cylinder_out")
            )
            intersection_points = ray_points + t[:, None] * ray_direction
            angles = torch.atan2(
                intersection_points[:, 2], intersection_points[:, 0])
            angles = torch.fmod(angles + 2 * np.pi, 2 * np.pi) / \
                (2 * np.pi) * (self.num_models // num_angle_sections)
            angles = torch.floor(angles)
            heights = torch.clamp(
                num_height_sections *
                (-intersection_points[:, 1] -
                 min_height) / (max_height - min_height),
                0, num_height_sections - 1)
            selection_indices = heights.long() * num_angle_sections + angles.long()
            selection_indices = torch.where(torch.all(torch.isfinite(
                intersection_points), dim=1), selection_indices, 0)
            selection_logits = torch.ones(
                (angles.shape[0], self.num_models), dtype=inputs.dtype, device=inputs.device)
            selection_probabilities = torch.softmax(selection_logits, dim=1)
        else:
            raise ValueError(
                f"Unknown model selection mode {self.selection_mode}")

        model_outputs = []
        for factor in factors:
            x = inputs
            x = self.model_layers[0](x, selection_indices, (factor[0], 1))
            x = self.model_layers[1](x)
            for i in range(1, self.layers - 1):
                my_factor = (factor[i % 2], factor[(i + 1) % 2])
                # my_subfactor = (factor[2 + i % 2], factor[2 + (i + 1) % 2])
                x = self.model_layers[2 * i](x, selection_indices, my_factor)
                x = self.model_layers[2 * i + 1](x)
            x = self.model_layers[2 * (self.layers - 1)](x,
                                                         selection_indices, (1, factor[self.layers % 2]))
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return MultiMipnetOutputs(
            model_outputs=model_outputs,
            selection_indices=selection_indices,
            selection_logits=selection_logits,
            selection_probabilities=selection_probabilities
        )

    def num_params(self, lod: int):
        factor = lod if isinstance(lod, Tuple) else self.factors[lod]
        total_params = 0
        if self.selection_model:
            total_params += sum([x.numel()
                                for x in self.selection_model.parameters()])
        total_params += self.model_layers[0].num_params((factor[0], 1))
        for i in range(1, self.layers - 1):
            my_factor = (factor[i % 2], factor[(i + 1) % 2])
            total_params += self.model_layers[2 * i].num_params(my_factor)
        x = self.model_layers[2 * (self.layers - 1)
                              ].num_params((1, factor[self.layers % 2]))
        return total_params

    def get_proto(self) -> model_pb2.Model.Network:
        """Get proto representation.

        Returns:
            Proto containing the network weights.
        """
        network = model_pb2.Model.Network()
        state_dict = self.state_dict()
        for k, v in state_dict.items():
            v_np = v.cpu().numpy()
            network.state_dict[k].shape.extend(v_np.shape)
            network.state_dict[k].values.extend(v_np.ravel())
        network.multisubnet.num_models = self.num_models
        network.multisubnet.in_features = self.in_features
        network.multisubnet.out_features = self.out_features
        network.multisubnet.layers = self.layers
        network.multisubnet.hidden_features = self.hidden_features
        network.multisubnet.selection_layers = self.selection_layers
        network.multisubnet.selection_hidden_features = self.selection_hidden_features
        network.multisubnet.selection_mode = self.selection_mode
        return network
