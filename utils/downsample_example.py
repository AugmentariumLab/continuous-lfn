import os
import subprocess

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import tqdm

import sat_utils


def downsample_circles():
    image_filename = "assets/concentric_circles.png"
    image = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
    print("image shape", image.shape)
    new_height = new_width = 64
    nearest_downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    area_downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # cv2.imwrite("assets/downsampled_nearest.png",
    #             nearest_downsampled_image)
    # cv2.imwrite("assets/downsampled_area.png", area_downsampled_image)
    nearest_downsampled_image = cv2.resize(
        nearest_downsampled_image, (256, 256), interpolation=cv2.INTER_NEAREST)
    area_downsampled_image = cv2.resize(
        area_downsampled_image, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("assets/downsampled_nearest_up.png",
                nearest_downsampled_image)
    cv2.imwrite("assets/downsampled_area_up.png", area_downsampled_image)
    exit(1)

    new_image = []
    mask_size = 10
    for channel in range(image.shape[2]):
        bgimage = image[:, :, channel]
        f = np.fft.fft2(bgimage)
        fshift = np.fft.fftshift(f)
        # magnitude_spectrum = 20 * np.log(np.abs(fshift))
        rows, cols = bgimage.shape
        crow, ccol = rows // 2, cols // 2
        plt.imsave(f"assets/fourier_{channel}.png", 20 *
                   np.log(np.abs(fshift) + 1e-10), cmap="gray")

        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - mask_size:crow + mask_size,
             ccol - mask_size:ccol + mask_size] = 1
        # apply mask and inverse DFT
        fshift = fshift * mask
        plt.imsave(f"assets/fourier_pass_{channel}.png",
                   20 * np.log(np.abs(fshift) + 1e-10), cmap="gray")

        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        new_image.append(img_back)
    new_image = np.stack(new_image, axis=-1)
    cv2.imwrite("assets/downsampled_dft.png", new_image)


def downsample_image():
    base_dir = "datasets/DavidDataset0113Config5v4/frames/"
    images = [
        "Board103Camera1.png",
        "Board112Camera2.png",
    ]
    images = [os.path.join(base_dir, x) for x in images]
    crops = [
        [1723, 492, 2342, 2012],
        [1723, 857, 2521, 2352]
    ]
    for i, image_filename in enumerate(images):
        image = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
        print("image shape", image.shape)
        crop = crops[i]
        for f in [1, 2, 4, 8]:
            if f != 1:
                area_downsampled_image = cv2.resize(
                    image, (0, 0), fx=1/f, fy=1/f, interpolation=cv2.INTER_AREA)
            else:
                area_downsampled_image = image.copy()
            crop_resized = [crop[0]//f, crop[1]//f, crop[2]//f, crop[3]//f]
            cropped_image = area_downsampled_image[crop_resized[1]:crop_resized[3], crop_resized[0]:crop_resized[2]]
            cv2.imwrite(f"temp/{i}_downsampled_area_f{f}.png", cropped_image)


def downsample_test():
    device = "cuda"
    scale = 8
    crop = np.array([1614, 1415, 2333, 2939])
    crop_resized = crop//scale
    input_image = "runs/run_2022_08_18_mipnetsat_jon_tvt/frames/level3/00010.png"
    input_dir = "runs/run_2022_08_18_mipnetsat_jon_tvt/frames/level3/"
    files = os.listdir(input_dir)
    files = sorted(files)
    nearest_output_dir = "temp/nearest"
    downsampled_output_dir = "temp/downsampled/"
    filtered_output_dir = "temp/filtered"
    filtered_sampled_dir = "temp/filtered_sampled"
    os.makedirs(nearest_output_dir, exist_ok=True)
    os.makedirs(downsampled_output_dir, exist_ok=True)
    os.makedirs(filtered_output_dir, exist_ok=True)
    os.makedirs(filtered_sampled_dir, exist_ok=True)
    for file in tqdm.tqdm(files):
        input_image = os.path.join(input_dir, file)
        image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
        nearest_image = cv2.resize(
            image, (0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(nearest_output_dir, file),
                    nearest_image[crop_resized[1]:crop_resized[3], crop_resized[0]:crop_resized[2]])
        downsampled_image = cv2.resize(
            image, (0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)
        height, width, _ = image.shape
        cv2.imwrite(os.path.join(downsampled_output_dir, file),
                    downsampled_image[crop_resized[1]:crop_resized[3], crop_resized[0]:crop_resized[2]])
        image_torch = torch.tensor(
            image / 255, dtype=torch.float32, device=device)
        sat = sat_utils.build_sat_image(image_torch[None], fp_sampling=True)
        xx, yy = torch.meshgrid(torch.linspace(-1, 1, width, dtype=torch.float32, device=device),
                                torch.linspace(-1, 1, height,
                                               dtype=torch.float32, device=device),
                                indexing="xy")
        z = torch.zeros_like(xx)
        xy_positions = torch.stack(
            (xx, yy, z), axis=-1).reshape(height*width, 3)
        filtered_image = sat_utils.sample_from_sat_fp(sat, xy_positions, scale)
        filtered_image = filtered_image.reshape(height, width, 4)
        filtered_image = (255 * filtered_image).cpu().numpy()
        cv2.imwrite(os.path.join(filtered_output_dir, file),
                    filtered_image[crop[1]:crop[3], crop[0]:crop[2]])
        filtered_sampled_image = cv2.resize(
            filtered_image, (0, 0), fx=1/8, fy=1/8, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(filtered_sampled_dir, file),
                    filtered_sampled_image[crop_resized[1]:crop_resized[3], crop_resized[0]:crop_resized[2]])
    subprocess.run(["ffmpeg", "-i", f"{nearest_output_dir}/%05d.png",
                    "-filter_complex", "split=2[bg][fg];[bg]drawbox=c=white@1:replace=1:t=fill[bg];[bg][fg]overlay=format=auto,split=2[v1][v2];[v1]palettegen=stats_mode=full[palette];[v2][palette]paletteuse=dither=sierra2_4a", "-y",
                    f"temp/nearest.gif"])
    subprocess.run(["ffmpeg", "-i", f"{downsampled_output_dir}/%05d.png",
                    "-filter_complex", "split=2[bg][fg];[bg]drawbox=c=white@1:replace=1:t=fill[bg];[bg][fg]overlay=format=auto,split=2[v1][v2];[v1]palettegen=stats_mode=full[palette];[v2][palette]paletteuse=dither=sierra2_4a", "-y",
                    f"temp/downsampled.gif"])
    subprocess.run(["ffmpeg", "-i", f"{filtered_output_dir}/%05d.png",
                    "-filter_complex", "split=2[bg][fg];[bg]drawbox=c=white@1:replace=1:t=fill[bg];[bg][fg]overlay=format=auto,split=2[v1][v2];[v1]palettegen=stats_mode=full[palette];[v2][palette]paletteuse=dither=sierra2_4a", "-y",
                    f"temp/filtered.gif"])
    subprocess.run(["ffmpeg", "-i", f"{filtered_sampled_dir}/%05d.png",
                    "-filter_complex", "split=2[bg][fg];[bg]drawbox=c=white@1:replace=1:t=fill[bg];[bg][fg]overlay=format=auto,split=2[v1][v2];[v1]palettegen=stats_mode=full[palette];[v2][palette]paletteuse=dither=sierra2_4a", "-y",
                    f"temp/sampled.gif"])


def downsample_vs_sat_filtering():
    image_filename = "person_not_exists_1.jpeg"
    image_basename = os.path.splitext(image_filename)[0]
    image_path = os.path.join("assets", image_filename)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width, channels = image.shape
    image = cv2.resize(
        image, (width//4, height//4), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"assets/{image_basename}_input.png",
                image)
    print("image shape", image.shape)
    height, width, channels = image.shape
    device = "cpu"
    scale = 4
    new_height = height // scale
    new_width = width // scale
    area_downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    cv2.imwrite(f"assets/{image_basename}_downsampled.png",
                area_downsampled_image)
    area_downsampled_image_up = cv2.resize(
        area_downsampled_image, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"assets/{image_basename}_downsampled_up.png",
                area_downsampled_image_up)

    image_torch = torch.tensor(image / 255, dtype=torch.float32, device=device)
    sat = sat_utils.build_sat_image(image_torch[None], fp_sampling=True)
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, width, dtype=torch.float32, device=device),
                            torch.linspace(-1, 1, height,
                                           dtype=torch.float32, device=device),
                            indexing="xy")
    z = torch.zeros_like(xx)
    xy_positions = torch.stack((xx, yy, z), axis=-1).reshape(height*width, 3)
    filtered_image = sat_utils.sample_from_sat_fp(sat, xy_positions, scale)
    filtered_image = filtered_image.reshape(height, width, 3).cpu().numpy()
    cv2.imwrite(
        f"assets/{image_basename}_sat_filtered.png", filtered_image*255)


if __name__ == "__main__":
    # downsample_circles()
    # downsample_image()
    # downsample_test()
    downsample_vs_sat_filtering()
