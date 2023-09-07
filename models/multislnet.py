"""Multiple networks with shared layers for multiple LODs."""
import dataclasses
from typing import Optional, Tuple, Union, Sequence
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import tqdm

from models import mlp
from models.multimodel import MultiModelLinear, MultiModelLayerNorm, SELECTION_MODES, AVAILABLE_SELECTION_MODES
from utils import my_torch_utils
from protos_compiled import model_pb2


@dataclasses.dataclass
class MultiSLNetOutputs:
    """Outputs for the multi model MLP.

    Attributes:
        outputs: The final outputs as (B, L, F) tensor.
        selection_indices: Indices for which model to use.
        selection_logits: Outputs for the selection model.
        selection_probabilities: Probabilities for which model to use.
    """
    model_outputs: torch.Tensor
    selection_indices: torch.Tensor
    selection_logits: torch.Tensor
    selection_probabilities: torch.Tensor


class MultiSLNet(nn.Module):
    def __init__(self, num_models: int = 64, in_features: int = 6, out_features: int = 3,
                 layers: int = 5, hidden_features: int = 64,
                 use_layernorm: bool = False,
                 selection_layers: int = 5, selection_hidden_features: int = 64,
                 lerp_value: float = 0.5, selection_mode: str = "angle",
                 clustering_feature_size: int = 0,
                 first_stage_epochs: int = 0, shared_first_layers: int = 2):
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
            first_stage_epochs: Number of epochs for two-stage training. If 0, no two-stage training is used.
            shared_first_layers: Number of initial layers that are shared between all models.
        """
        super().__init__()
        if num_models <= 0:
            raise ValueError(f"Num models is <= 0: {num_models}")
        self.num_models = num_models
        self.in_features = in_features
        self.out_features = out_features
        self.layers = layers
        self.hidden_features = hidden_features
        self.use_layernorm = use_layernorm
        self.selection_layers = selection_layers
        self.selection_hidden_features = selection_hidden_features
        self.selection_mode = selection_mode
        self.lerp_value = lerp_value
        self.num_angle_sections = 8
        self.num_height_sections = 8
        self.max_height = 1.0
        self.min_height = -3.0
        self.radius = 1.0
        self.clustering_feature_size = clustering_feature_size
        self.num_outputs = layers - shared_first_layers + 1
        print(f"{self.__class__.__name__} Number of LODS: {self.num_outputs}")
        self.first_stage_epochs = first_stage_epochs
        self.shared_first_layers = shared_first_layers
        self.current_epoch = nn.Parameter(torch.tensor(0), requires_grad=False)

        if selection_mode not in AVAILABLE_SELECTION_MODES:
            print("Supported selection modes: ", AVAILABLE_SELECTION_MODES)
            raise ValueError(f"Selection mode {selection_mode} not supported.")

        if self.selection_mode == SELECTION_MODES.MLP.value:
            self.selection_model = mlp.MLP(in_features=in_features, out_features=num_models, layers=selection_layers,
                                           hidden_features=selection_hidden_features)
        num_models_in_layer = (num_models+1) if shared_first_layers == 0 else 1
        model_layers = [MultiModelLinear(
            num_models=num_models_in_layer,
            in_features=in_features,
            out_features=hidden_features
        )]
        if use_layernorm:
            model_layers.append(MultiModelLayerNorm(
                num_models=num_models_in_layer,
                normalized_shape=hidden_features
            ))
        model_layers.append(nn.ReLU())
        for i in range(1, layers - 1):
            num_models_in_layer = (
                num_models+1) if i >= shared_first_layers else 1
            model_layers.append(MultiModelLinear(
                num_models=num_models_in_layer,
                in_features=hidden_features,
                out_features=hidden_features
            ))
            if use_layernorm:
                model_layers.append(MultiModelLayerNorm(
                    num_models=num_models_in_layer,
                    normalized_shape=hidden_features
                ))
            model_layers.append(nn.ReLU())
        num_models_in_layer = (
            num_models+1) if layers >= shared_first_layers else 1
        model_layers.append(MultiModelLinear(
            num_models=num_models_in_layer,
            in_features=hidden_features,
            out_features=out_features
        ))
        self.model_layers = nn.ModuleList(model_layers)
        if clustering_feature_size > 0:
            self.clustering_features = nn.Parameter(torch.rand(
                (num_models, clustering_feature_size), dtype=torch.float32))

    def freeze_selection_model(self):
        if hasattr(self, "selection_model"):
            for parameter in self.selection_model.parameters():
                parameter.requires_grad = False

    def set_epoch(self, epoch: int):
        if isinstance(self.current_epoch, nn.Parameter):
            self.current_epoch.data = torch.tensor(
                epoch, device=self.current_epoch.device)
        else:
            self.current_epoch = epoch
        if self.first_stage_epochs > 0 and self.current_epoch == self.first_stage_epochs:
            self.initialize_second_stage()

    def initialize_second_stage(self):
        tqdm.tqdm.write("Initializing second stage")
        for layer in self.model_layers:
            if isinstance(layer, MultiModelLinear) or isinstance(layer, MultiModelLayerNorm):
                layer.repeat_first_model_weights()

    def forward_through_layers(self, inputs, indices, num_shared_layers: int):
        x = inputs
        if self.use_layernorm:
            for i in range(self.layers - 1):
                indices_for_layer = indices if i >= num_shared_layers else 0
                x = self.model_layers[3 * i](x, indices_for_layer)
                x = self.model_layers[3 * i + 1](x, indices_for_layer)
                x = self.model_layers[3 * i + 2](x)
            indices_for_layer = indices if self.layers - 1 >= num_shared_layers else 0
            x = self.model_layers[3 * (self.layers - 1)](x, indices_for_layer)
        else:
            for i in range(self.layers - 1):
                indices_for_layer = indices if i >= num_shared_layers else 0
                x = self.model_layers[2 * i](x, indices_for_layer)
                x = self.model_layers[2 * i + 1](x)
            indices_for_layer = indices if self.layers - 1 >= num_shared_layers else 0
            x = self.model_layers[2 * (self.layers - 1)](x, indices_for_layer)
        return x

    def forward_first_stage(self, inputs: torch.Tensor, lods: Optional[Sequence[Union[int, float]]] = None) -> MultiSLNetOutputs:
        selection_logits = torch.cat((
            torch.zeros((1,), dtype=inputs.dtype, device=inputs.device),
            torch.full((self.num_models - 1,), -999.9,
                       dtype=inputs.dtype, device=inputs.device)
        ), dim=0)
        selection_logits = selection_logits[None, :].expand(
            inputs.shape[0], -1)
        selection_probabilities = torch.cat((
            torch.ones((1,), dtype=inputs.dtype, device=inputs.device),
            torch.zeros((self.num_models - 1,),
                        dtype=inputs.dtype, device=inputs.device)
        ), dim=0)
        selection_probabilities = selection_probabilities[None, :].expand(
            inputs.shape[0], -1)
        selection_indices = torch.zeros_like(
            inputs[:, 0], dtype=torch.long)
        model_outputs = []
        for lod in lods:
            num_shared_layers = self.layers - lod
            x = self.forward_through_layers(inputs, 0, num_shared_layers)
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return MultiSLNetOutputs(
            model_outputs=model_outputs,
            selection_indices=selection_indices,
            selection_logits=selection_logits,
            selection_probabilities=selection_probabilities
        )

    def forward(self, inputs: torch.Tensor, lods: Optional[Sequence[Union[int, float]]] = None) -> MultiSLNetOutputs:
        """Forward pass.

        Args:
            inputs: Inputs as (B, F) tensor.

        Returns:
            The model outputs as (B, F) tensor.
        """
        if lods is None:
            # lods = [0, self.layers - self.shared_first_layers]
            lods = np.arange(self.layers - self.shared_first_layers + 1)
        if self.first_stage_epochs > 0 and self.current_epoch < self.first_stage_epochs and self.training:
            return self.forward_first_stage(inputs, lods)
        if self.selection_mode == SELECTION_MODES.ANGLE.value:
            angles = torch.atan2(inputs[:, 2], inputs[:, 0])
            angles = torch.fmod(angles + 2 * np.pi, 2 *
                                np.pi) / (2 * np.pi) * self.num_models
            angles = torch.floor(angles)
            selection_indices = angles.long()
            selection_logits = torch.ones(
                (angles.shape[0], self.num_models), dtype=inputs.dtype, device=inputs.device)
            selection_probabilities = torch.softmax(selection_logits, dim=1)
        elif self.selection_mode == SELECTION_MODES.MLP.value:
            selection_logits = self.selection_model(inputs)
            selection_probabilities = torch.softmax(selection_logits, dim=1)
            if False and self.training:
                selection_probabilities = (
                    1 - self.lerp_value) + self.lerp_value * selection_probabilities
                selection_indices = torch.multinomial(
                    selection_probabilities, 1)[:, 0]
            else:
                selection_indices = torch.argmax(
                    selection_probabilities, dim=1)
        elif self.selection_mode in (SELECTION_MODES.CYLINDER.value, SELECTION_MODES.CYLINDER_OUT.value):
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
                use_minus=(self.selection_mode == SELECTION_MODES.CYLINDER_OUT)
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
        for lod in lods:
            num_shared_layers = self.layers - lod
            x = self.forward_through_layers(inputs, selection_indices+1, num_shared_layers)
            model_outputs.append(x)
        model_outputs = torch.stack(model_outputs, dim=1)
        return MultiSLNetOutputs(
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
        network.multimodel.num_models = self.num_models
        network.multimodel.in_features = self.in_features
        network.multimodel.out_features = self.out_features
        network.multimodel.layers = self.layers
        network.multimodel.hidden_features = self.hidden_features
        network.multimodel.selection_layers = self.selection_layers
        network.multimodel.selection_hidden_features = self.selection_hidden_features
        network.multimodel.selection_mode = self.selection_mode
        return network

    def num_params(self, lod: int) -> int:
        total_params = 0
        if hasattr(self, "selection_model"):
            total_params += self.selection_model.num_params(lod)
        for layer in self.model_layers:
            total_params += sum(p.numel() for p in layer.parameters())
        total_params += self.num_models * self.clustering_feature_size
        return int(total_params)
