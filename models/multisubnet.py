"""Wrapper model which consists of multiple submodels."""
import dataclasses
from typing import Tuple

import torch
from torch import nn
import numpy as np

from models import mlp

from protos_compiled import model_pb2


def solve_quadratic(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Solution to quadratic formula."""
    return (-b - torch.sqrt(b * b - 4 * a * c)) / (2 * a)


@dataclasses.dataclass
class MultiSubnetOutputs:
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


class MultiSubnetLinear(nn.Module):
    def __init__(self, num_models: int, in_features: int, out_features: int):
        super().__init__()
        sqrt_k = np.sqrt(1 / num_models)

        self.weights = nn.Parameter(
            2 * sqrt_k * torch.rand((num_models, out_features, in_features), dtype=torch.float32) - sqrt_k)
        self.biases = nn.Parameter(
            2 * sqrt_k * torch.rand((num_models, out_features), dtype=torch.float32) - sqrt_k)

    def forward(self, inputs: torch.Tensor, indices: torch.Tensor, factors: Tuple[int, int]):
        weight = self.weights
        bias = self.biases
        subweight = weight[indices, :weight.shape[1] //
                           factors[0], :weight.shape[2] // factors[1]]
        subbias = bias[indices, :weight.shape[1] // factors[0]]
        return torch.matmul(subweight, inputs[:, :, None])[:, :, 0] + subbias


class MultiSubnet(nn.Module):
    def __init__(self, num_models: int = 64, in_features: int = 6, out_features: int = 3,
                 layers: int = 5, hidden_features: int = 64,
                 selection_layers: int = 5, selection_hidden_features: int = 64,
                 lerp_value: float = 0.5, selection_mode: str = "angle"):
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
        model_layers = [MultiSubnetLinear(
            num_models=num_models,
            in_features=in_features,
            out_features=hidden_features
        ), nn.ReLU()]
        for layer in range(layers - 2):
            model_layers.append(MultiSubnetLinear(
                num_models=num_models,
                in_features=hidden_features,
                out_features=hidden_features
            ))
            model_layers.append(nn.ReLU())
        model_layers.append(MultiSubnetLinear(
            num_models=num_models,
            in_features=hidden_features,
            out_features=out_features
        ))
        self.model_layers = nn.ModuleList(model_layers)

    def freeze_selection_model(self):
        if self.selection_model is not None:
            for parameter in self.selection_model.parameters():
                parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor, factors=None) -> MultiSubnetOutputs:
        """Forward pass.

        Args:
            inputs: Inputs as (B, F) tensor.

        Returns:
            The model outputs as (B, F) tensor.
        """
        if factors is None:
            factors = [
                (2, 2),
                (1, 2),
                (1, 1)
            ]
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
        elif self.selection_mode == "cylinder":
            num_angle_sections = self.num_angle_sections
            num_height_sections = self.num_height_sections
            max_height = self.max_height
            min_height = self.min_height
            radius = self.radius
            assert num_angle_sections * num_height_sections >= self.num_models, "Invalid"
            ray_direction = inputs[:, :3]
            ray_points = torch.cross(inputs[:, 3:6], inputs[:, :3])
            t = solve_quadratic(
                ray_direction[:, [0, 2]].pow(2).sum(1),
                (2 * ray_points[:, [0, 2]] * ray_direction[:, [0, 2]]).sum(-1),
                ray_points[:, [0, 2]].pow(2).sum(1) - radius * radius
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
                x = self.model_layers[2 * i](x, selection_indices, my_factor)
                x = self.model_layers[2 * i + 1](x)
            x = self.model_layers[2 * (self.layers - 1)](x,
                                                         selection_indices, (1, factor[self.layers % 2]))
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return MultiSubnetOutputs(
            model_outputs=model_outputs,
            selection_indices=selection_indices,
            selection_logits=selection_logits,
            selection_probabilities=selection_probabilities
        )

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
