import cv2
import numpy as np
from scipy import ndimage
import torch


def compute_si(frame):
    sobel_image = ndimage.sobel(frame)
    si = np.mean(np.abs(sobel_image))
    return si


def compute_ti(previous_frame, frame):
    diff_frame = frame - previous_frame
    ti = np.mean(np.abs(diff_frame))
    return ti


def compute_fft_flicker(prev_frame, prev_reference, current_frame, current_reference):
    assert prev_frame.max() <= 1.0

    height, width = prev_frame.shape[:2]
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    radius = np.sqrt(np.power(xx - width / 2, 2.0) +
                     np.power(yy - height / 2, 2.0))
    radius_int = radius.astype(np.int)
    max_frequency = np.max(radius)
    frequency_low = int(0.01 * max_frequency)
    frequency_mid = int(0.16 * max_frequency)
    frequency_high = int(0.80 * max_frequency)

    prev_frame_gray = cv2.cvtColor(
        (prev_frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    prev_reference_gray = cv2.cvtColor(
        (prev_reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    current_frame_gray = cv2.cvtColor(
        (current_frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    current_reference_gray = cv2.cvtColor(
        (current_reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0

    # si = compute_si(current_reference_gray)
    # ti = compute_ti(prev_reference_gray, current_reference_gray)
    # m = max(si * ti, 0.007)

    d_t = current_frame_gray - current_reference_gray
    d_t_1 = prev_frame_gray - prev_reference_gray
    change = d_t - d_t_1
    f = np.fft.fft2(change)
    fshift = np.fft.fftshift(f)
    f_binned = np.bincount(radius_int.ravel(), np.abs(fshift).ravel())

    # plt.imsave("gray_frame.png", gray_frame, cmap="gray")
    # plt.imsave("previous_frame.png", previous_frame, cmap="gray")
    # plt.imsave("ref_gray_frame.png", ref_gray_frame, cmap="gray")
    # plt.imsave("previous_ref_frame.png", previous_ref_frame, cmap="gray")
    # plt.imsave("dt.png", np.abs(d_t), vmin=0, vmax=0.1)
    # plt.imsave("dt1.png", np.abs(d_t_1), vmin=0, vmax=0.1)
    # plt.imsave("change.png", np.abs(change), vmin=0, vmax=0.1)
    # plt.plot(f_binned)
    # plt.savefig("fft.png")
    # exit(0)

    nr = np.bincount(radius_int.ravel())
    f_binned = f_binned / nr
    s_l = np.sum(f_binned[frequency_low:frequency_mid + 1]
                 ) / (frequency_mid - frequency_low)
    s_h = np.sum(f_binned[frequency_mid:frequency_high + 1]
                 ) / (frequency_high - frequency_mid)
    frame_flicker = (s_l + s_h)
    return frame_flicker


def compute_fft_flicker_torch(prev_frame, prev_reference, current_frame, current_reference):
    assert prev_frame.max() <= 1.0

    height, width = prev_frame.shape[:2]
    xx, yy = torch.meshgrid(torch.arange(width, device=prev_frame.device, dtype=torch.float32),
                            torch.arange(
                                height, device=prev_frame.device, dtype=torch.float32),
                            indexing="xy")
    radius = torch.sqrt(torch.pow(xx - width / 2, 2.0) +
                        torch.pow(yy - height / 2, 2.0))
    radius_int = radius.int()
    max_frequency = torch.max(radius)
    frequency_low = int(0.01 * max_frequency)
    frequency_mid = int(0.16 * max_frequency)
    frequency_high = int(0.80 * max_frequency)

    prev_frame_gray = (prev_frame[:, :, 0] * 0.299 +
                       prev_frame[:, :, 1] * 0.587 + prev_frame[:, :, 2] * 0.114)
    prev_reference_gray = (prev_reference[:, :, 0] * 0.299 +
                           prev_reference[:, :, 1] * 0.587 + prev_reference[:, :, 2] * 0.114)
    current_frame_gray = (current_frame[:, :, 0] * 0.299 +
                          current_frame[:, :, 1] * 0.587 + current_frame[:, :, 2] * 0.114)
    current_reference_gray = (current_reference[:, :, 0] * 0.299 +
                              current_reference[:, :, 1] * 0.587 + current_reference[:, :, 2] * 0.114)

    d_t = current_frame_gray - current_reference_gray
    d_t_1 = prev_frame_gray - prev_reference_gray
    change = d_t - d_t_1
    f = torch.fft.fft2(change)
    fshift = torch.fft.fftshift(f)
    f_binned = torch.bincount(radius_int.view(-1), torch.abs(fshift).view(-1))

    nr = torch.bincount(radius_int.view(-1))
    f_binned = f_binned / nr
    s_l = torch.sum(f_binned[frequency_low:frequency_mid + 1]
                    ) / (frequency_mid - frequency_low)
    s_h = torch.sum(f_binned[frequency_mid:frequency_high + 1]
                    ) / (frequency_high - frequency_mid)
    frame_flicker = (s_l + s_h)
    return frame_flicker
