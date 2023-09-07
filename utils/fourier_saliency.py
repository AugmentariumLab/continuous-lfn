# Estimate saliency using image differences. (idea from Brandon).
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

import cv2
import numpy as np
import tqdm

def compute_saliency_image(image, mask_size = 400):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image_grayscale)
    fshift = np.fft.fftshift(f)
    height, width, _ = image.shape
    mask = np.zeros((height, width), np.int32)
    crow, ccol = height // 2, width // 2
    mask[(crow - mask_size):(crow + mask_size),
            (ccol - mask_size):(ccol + mask_size)] = 1
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    new_image = np.real(img_back)
    delta_image = np.abs(image_grayscale - new_image)
    delta_image = (3 * delta_image).clip(0, 255)
    return delta_image

def load_and_process_image(input_image_path, mask_size):
    image = cv2.imread(input_image_path)
    saliency_image = compute_saliency_image(image, mask_size=mask_size)
    output_image_path = input_image_path.replace("com", "fourier_saliency").replace(".png", ".jpg")
    # print("output_image_path", output_image_path)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, saliency_image)

def estimate_fourier_saliency():
    dataset = "StudioV2CaptureTest0805_max_light"
    base_dir = "datasets"
    mask_size = 75
    
    # image_path = os.path.join(base_dir, dataset, 
    #     "DespilledFrames", "Board121Camera4", 
    #     "com", "00000.png")
    # load_and_process_image(image_path, mask_size)
    
    image_path_glob =os.path.join(base_dir, dataset, 
        "DespilledFrames", "*", "com", "00000.png")
    image_paths = glob.glob(image_path_glob)
    executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    my_futures = []
    for image_path in image_paths:
        executor.submit(load_and_process_image, image_path, mask_size)
    with tqdm.tqdm(total=len(my_futures)) as pbar:
        for future in as_completed(my_futures):
            future.result()
            pbar.update(1)
    executor.shutdown()

if __name__ == "__main__":
    estimate_fourier_saliency()