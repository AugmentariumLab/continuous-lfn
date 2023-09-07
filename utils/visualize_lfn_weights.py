import os

import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import my_utils

def visualize_lfn_weights(self):
    model = self.model
    weight_size = 512
    output_directory = my_utils.join_and_make(self.args.checkpoints_dir, 'weights_visualization')
    # Get the weights of each layer
    for name, param in model.named_parameters():
        print("Name: ", name)
        if 'weight' in name and len(param.data.shape) == 2:
            weights = param.data.cpu().numpy()
            num_neurons = weights.shape[0]
            num_inputs = weights.shape[1]
            # Pad the weights with zeros to make them square
            # weights = np.pad(weights, ((0, weight_size - num_neurons), (0, weight_size - num_inputs)), 'constant')
            # Upsample the weights by a factor of 4
            upscale_factor = 8
            weights = weights.repeat(upscale_factor, axis=0).repeat(upscale_factor, axis=1)
            # Normalize the weights to be between 0 and 1
            weights_normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
            # Convert the weights to a colormap
            weights_turbo_index = weights_normalized * (len(my_utils.TURBO_COLORMAP) - 1)
            weights_turbo_floored = np.floor(weights_turbo_index)
            weights_turbo_ceil = np.ceil(weights_turbo_index)
            weights_turbo_delta = weights_turbo_index - weights_turbo_floored

            weights_turbo = my_utils.lerp(
                my_utils.TURBO_COLORMAP[weights_turbo_floored.ravel().astype(int)],
                my_utils.TURBO_COLORMAP[weights_turbo_ceil.ravel().astype(int)],
                np.tile(weights_turbo_delta.ravel()[:, np.newaxis], [1, 3]))
            weights_turbo = weights_turbo.reshape(weights.shape[0], weights.shape[1], 3)

            # Plot the weights and save them to a file
            save_path = os.path.join(output_directory, f"{name}.png")
            plt.imsave(save_path, weights)
            # plt.imsave(save_path, weights_turbo)
            