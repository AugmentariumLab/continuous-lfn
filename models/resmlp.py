"""This contains a basic MLP with layernorm and relu"""
import torch
from torch import nn

from protos_compiled import model_pb2


class ResMLPBlock(nn.Module):
    def __init__(self, features, use_layernorm=True):
        super().__init__()
        if use_layernorm:
            self.block = nn.Sequential(
                nn.Linear(features, features),
                nn.LayerNorm(features),
                nn.ReLU(),
                nn.Linear(features, features),
                nn.LayerNorm(features),
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(features, features),
                nn.ReLU(),
                nn.Linear(features, features),
            )
    
    def forward(self, x):
        return x + self.block(x)

class ResMLP(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3,
                 layers: int = 6, hidden_features: int = 256,
                 use_layernorm: bool = True):
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
            for _ in range((layers - 2)//2):
                model_layers.append(ResMLPBlock(hidden_features, use_layernorm))
                model_layers.append(nn.ReLU())
            if layers % 2 != 0:
                print("Warning: odd number of layers, last layer will be linear")
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

    def forward(self, inputs) -> torch.Tensor:
        return self.model(inputs)

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

    def num_params(self, lod: int) -> int:
        return sum(self._numel_linear(layer) for layer in self.model if isinstance(layer, nn.Linear))