import torch
import torch.nn.functional as F
import numpy as np


def build_sat_image(image: torch.Tensor, fp_sampling: bool):
    """Builds a summed area table from an image.

    Args:
        image (torch.Tensor): Image to build SAT from as (B, H, W, C) tensor.
        fp_sampling (bool): Whether to prepare the SAT for floating point sampling.

    Returns:
        torch.Tensor: SAT of image as (B, H, W, C) tensor.
    """
    x = torch.cumsum(image, dim=1)
    x = torch.cumsum(x, dim=2)
    if fp_sampling:
        return x.permute((3, 0, 1, 2))[None]
    return x


def sample_from_sat(sat: torch.Tensor, xy_positions: torch.Tensor, scale: float):
    """Samples from a summed area table.

    Args:
      sat: summed area table as (B, H, W, C) tensor.
      xy_positions: Positions to sample as (B, xyz) tensor.
      scale: which scale to sample (1 to infty).

    Returns:
      Sampled values as (B, RGBA) tensor.
    """
    dtype = sat.dtype
    device = sat.device
    half_scale = scale / 2 + 1e-3  # Avoid rounding errors.
    num_images, height, width, channels = sat.shape
    batch_size, _ = xy_positions.shape
    x = ((1+xy_positions[:, 0]) / 2) * (width-1)
    y = ((1+xy_positions[:, 1]) / 2) * (height-1)
    z = ((1+xy_positions[:, 2]) / 2) * (num_images-1)
    z = torch.round(z).long()
    x_low = torch.round(x - half_scale).clamp(-1, width - 1).long()
    x_high = torch.round(x + half_scale).clamp(0, width - 1).long()
    y_low = torch.round(y - half_scale).clamp(-1, height - 1).long()
    y_high = torch.round(y + half_scale).clamp(0, height - 1).long()
    x_valid = (x_low >= 0)
    y_valid = (y_low >= 0)
    xy_valid = x_valid & y_valid
    area = (x_high - x_low) * (y_high - y_low)
    top_left = torch.zeros((batch_size, channels), dtype=dtype, device=device)
    top_left[xy_valid] = sat[z[xy_valid], y_low[xy_valid], x_low[xy_valid]]
    top_right = torch.zeros((batch_size, channels), dtype=dtype, device=device)
    top_right[y_valid] = sat[z[y_valid], y_low[y_valid], x_high[y_valid]]
    bottom_left = torch.zeros((batch_size, channels),
                              dtype=dtype, device=device)
    bottom_left[x_valid] = sat[z[x_valid], y_high[x_valid], x_low[x_valid]]
    bottom_right = sat[z, y_high, x_high]
    # print(f"shapes: {top_left.shape}, {top_right.shape}, {bottom_left.shape}, {bottom_right.shape}")
    # print(f"area: {area.shape}")
    values = (bottom_right - top_right -
              bottom_left + top_left) / area[:, None]
    return values


def sample_from_sat_fp(sat: torch.Tensor, xy_positions: torch.Tensor, scale: float):
    """Samples from a summed area table.

    Args:
      sat: summed area table as (0, C, B, H, W) tensor.
      xy_positions: Positions to sample as (B, xyz) tensor.
      scale: which scale to sample (1 to infty).

    Returns:
      Sampled values as (B, RGBA) tensor.
    """
    half_scale = scale / 2
    _, _, _, height, width = sat.shape
    x = ((1+xy_positions[:, 0]) / 2) * (width-1)
    y = ((1+xy_positions[:, 1]) / 2) * (height-1)
    z = xy_positions[:, 2]
    x_low = (x - half_scale - 0.5).clamp(-1, width - 1)
    x_high = (x + half_scale - 0.5).clamp(0, width - 1)
    y_low = (y - half_scale - 0.5).clamp(-1, height - 1)
    y_high = (y + half_scale - 0.5).clamp(0, height - 1)
    area = (x_high - x_low) * (y_high - y_low)
    x_low = x_low * (2 / (width - 1)) - 1
    x_high = x_high * (2 / (width - 1)) - 1
    y_low = y_low * (2 / (height - 1)) - 1
    y_high = y_high * (2 / (height - 1)) - 1
    top_left_indices = torch.stack((x_low, y_low, z), dim=1)[
        None, :, None, None, :]
    top_left = F.grid_sample(sat, top_left_indices,
                             mode="bilinear", align_corners=True)
    top_left = top_left[0, :, :, 0, 0].permute((1, 0))
    top_right_indices = torch.stack((x_high, y_low, z), dim=1)[
        None, :, None, None, :]
    top_right = F.grid_sample(sat, top_right_indices,
                              mode="bilinear", align_corners=True)
    top_right = top_right[0, :, :, 0, 0].permute((1, 0))
    bottom_left_indices = torch.stack((x_low, y_high, z), dim=1)[
        None, :, None, None, :]
    bottom_left = F.grid_sample(sat, bottom_left_indices,
                                mode="bilinear", align_corners=True)
    bottom_left = bottom_left[0, :, :, 0, 0].permute((1, 0))
    bottom_right_indices = torch.stack((x_high, y_high, z), dim=1)[
        None, :, None, None, :]
    bottom_right = F.grid_sample(sat, bottom_right_indices,
                                  mode="bilinear", align_corners=True)
    bottom_right = bottom_right[0, :, :, 0, 0].permute((1, 0))
    values = (bottom_right - top_right -
              bottom_left + top_left) / area[:, None]
    return values
