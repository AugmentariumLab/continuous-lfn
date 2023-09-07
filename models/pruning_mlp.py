"""This contains a basic MLP with layernorm and relu"""
from typing import Optional

import torch
from torch import nn

from utils import torch_checkpoint_manager
from protos_compiled import model_pb2


class PruningMLP(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 6, hidden_features: int = 256,
                 use_layernorm: bool = True,
                 load_from: Optional[str] = None):
        """Initialize an MLP.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            layers: Number of layers.
            hidden_features: Number of hidden features.
            use_layernorm: Whether to include layer norm.
        """
        super().__init__()
        if layers == 1:
            model_layers = [nn.Linear(in_features=in_features,
                                      out_features=out_features)]
        else:
            model_layers = [nn.Linear(in_features=in_features,
                                      out_features=hidden_features)]
            if use_layernorm:
                model_layers.append(nn.LayerNorm(hidden_features))
            model_layers.append(nn.ReLU())
            for i in range(layers - 2):
                model_layers.append(nn.Linear(in_features=hidden_features,
                                              out_features=hidden_features))
                if use_layernorm:
                    model_layers.append(nn.LayerNorm(hidden_features))
                model_layers.append(nn.ReLU())
            model_layers.append(
                nn.Linear(in_features=hidden_features, out_features=out_features))
        self.model = nn.Sequential(*model_layers)
        self.in_features = in_features
        self.out_features = out_features
        self.use_layernorm = use_layernorm
        self.hidden_features = hidden_features
        self.layers = layers
        self.num_outputs = 1
        self.accum_fisher_info = {}
        self.accum_fisher_count = 0
        self.weight_masks = {}

        for n, p in self.named_parameters():
            if p.requires_grad:
                self.accum_fisher_info[n] = p.detach().clone().zero_().cuda()

        if load_from:
            checkpoint_manager = torch_checkpoint_manager.CheckpointManager(
                load_from)
            latest_checkpoint = checkpoint_manager.load_latest_checkpoint()
            if latest_checkpoint is None:
                raise ValueError("Load from checkpoint not available")
            subnet_state_dict = latest_checkpoint['model_state_dict']
            for i, layer in enumerate(self.model):
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm):
                    layer.weight.data.copy_(
                        subnet_state_dict[f"model.{i}.weight"])
                    layer.bias.data.copy_(subnet_state_dict[f"model.{i}.bias"])

    def forward(self, inputs) -> torch.Tensor:
        x = inputs
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                layer_weight_key = f"model.{i}.weight"
                weight = layer.weight
                bias = layer.bias
                if layer_weight_key in self.weight_masks:
                    weight = weight * self.weight_masks[layer_weight_key].type(weight.dtype)
                x = torch.matmul(x, weight.T) + bias[None, :]
            elif isinstance(layer, nn.LayerNorm):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
        return x

    def accumulate_fisher(self) -> None:
        self.accum_fisher_count += 1
        for n, p in self.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    self.accum_fisher_info[n] += p.grad.detach() ** 2

    def reset_pruning(self) -> None:
        self.weight_masks.clear()

    def prune_node(self, layer: int, nodes: int, prune_nodes=True) -> None:
        """Prune the model.

        Args:
            layer: Layer to prune.
            nodes: Number of nodes to prune.
            prune_nodes: Whether to prune nodes or weights.
        """
        # Find the node with the smallest fisher information.
        layer_num = 0
        layer_ct = 0
        # Find the linear layer.
        for i, layer_i in enumerate(self.model):
            if isinstance(layer_i, nn.Linear):
                layer_num = i
                layer_ct += 1
            if layer_ct > layer:
                break
        print("Layer:", layer_num)
        weight_key = f"model.{layer_num}.weight"
        bias_key = f"model.{layer_num}.bias"
        fisher_weight = self.accum_fisher_info[weight_key]
        if prune_nodes:
            fisher_weight_means = torch.mean(fisher_weight, dim=1)
            fisher_weight_argmin = torch.argsort(fisher_weight_means)
            if weight_key not in self.weight_masks:
                self.weight_masks[weight_key] = torch.ones_like(
                    self.model[layer_num].weight)
            self.weight_masks[weight_key][fisher_weight_argmin[:nodes], :] = 0
            print("number of weights removed:", torch.sum(
                self.weight_masks[weight_key] == 0))
        else:
            # Prune weights.
            num_weights = nodes * self.model[layer_num].weight.shape[1]
            fisher_weight_argmin = torch.argsort(fisher_weight.ravel())
            if weight_key not in self.weight_masks:
                self.weight_masks[weight_key] = torch.ones_like(
                    self.model[layer_num].weight)
            self.weight_masks[weight_key].ravel(
            )[fisher_weight_argmin[:num_weights]] = 0
            print("number of weights removed:", torch.sum(
                self.weight_masks[weight_key] == 0))

    def get_proto(self) -> model_pb2.Model.Network:
        network = model_pb2.Model.Network()
        state_dict = self.state_dict()
        for k, v in state_dict.items():
            v_np = v.cpu().numpy()
            network.state_dict[k].shape.extend(v_np.shape)
            network.state_dict[k].values.extend(v_np.ravel())
        network.mlp.in_features = self.in_features
        network.mlp.out_features = self.out_features
        network.mlp.layers = self.layers
        network.mlp.hidden_features = self.hidden_features
        network.mlp.use_layernorm = self.use_layernorm
        return network

    def _numel_linear(self, layer: nn.Linear) -> int:
        return layer.weight.numel() + layer.bias.numel()

    def _numel_layernorm(self, layer: nn.LayerNorm) -> int:
        if layer.elementwise_affine:
            return layer.weight.numel() + layer.bias.numel()
        return 0

    def num_params(self, lod: int) -> int:
        total_params = 0
        total_params += self._numel_linear(self.model[0])
        total_params += self._numel_layernorm(self.model[1])
        for i in range(1, self.layers - 1):
            total_params += self._numel_linear(self.model[3 * i])
            total_params += self._numel_layernorm(self.model[3 * i + 1])
        total_params += self._numel_linear(self.model[-1])
        return int(total_params)
