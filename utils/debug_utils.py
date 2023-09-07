import torch
import cv2
import numpy as np

from utils import sat_utils

def test_sat_sampling(app, height, width, image_sat, data):
    device = app.device
    args = app.args

    # Test SAT sampling.
    # my_torch_utils.save_torch_image(f"temp/gt_image.png", data["image"][0][:, :, :3])
    u = torch.linspace(-1, 1, width).to(device)
    v = torch.linspace(-1, 1, height).to(device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    uvs = torch.stack([uu, vv, torch.zeros_like(uu)-1], dim=-1)
    uvs = uvs.reshape(height * width, 3)
    lod_ray_batch = sat_utils.sample_from_sat_fp(
            image_sat, uvs, 1)
    sample_delta = (lod_ray_batch - data["image"][0].reshape(height*width, 4).to(device)).abs()
    sample_delta_max = sample_delta.max()
    sample_delta_mean = sample_delta.mean()
    print("sample delta:", sample_delta_max, sample_delta_mean)

    for j, scale in enumerate(np.linspace(1, 12, 100)):
        if args.sat_sample_floatingpoint:
            lod_ray_batch = sat_utils.sample_from_sat_fp(
                image_sat, uvs, scale)
        else:
            exit(1)
            lod_ray_batch = sat_utils.sample_from_sat(
                image_sat, uvs.reshape(-1, 3), scale)
        print("lod_ray_batch.shape", lod_ray_batch.shape)
        lod_ray_batch = lod_ray_batch.reshape(height, width, 4)
        print("lod_ray_batch.shape", lod_ray_batch.shape)
        rescaled_image = lod_ray_batch.cpu().numpy()
        crop=(1276, 726,2096, 2439)
        rescaled_image = rescaled_image[crop[1]:crop[3], crop[0]:crop[2], :]
        cropped_height, cropped_width, _ = rescaled_image.shape
        rescaled_image = cv2.cvtColor(rescaled_image, cv2.COLOR_RGBA2BGRA)
        # my_torch_utils.save_torch_image(f"temp/lod_ray_batch_{j}.png", lod_ray_batch)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        color = (0, 0, 0, 1)
        font_thickness = 5
        text = f"Scale 1/{scale:.4f}"
        text_size, _ = cv2.getTextSize(
            text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        org = (int(0.5 * cropped_width - 0.5 * text_w),
                int(cropped_height-0.5 * text_h))
        rescaled_image = cv2.putText(rescaled_image, text, org, font,
                            font_scale, color, font_thickness, cv2.LINE_AA)
        cv2.imwrite(f"temp/lod_ray_batch_{j}.png", 255 * rescaled_image)
    exit(0)