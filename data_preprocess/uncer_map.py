import cv2
import numpy as np
import os
import imageio
from tqdm import tqdm

def generate_uncertainty_map(image_paths):
    # Load the images
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    std = np.std(images, axis=0)

    std_normalized = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX)

    std_normalized = std_normalized.astype(np.uint8)

    # Invert the image to get the uncertainty map
    # uncertainty_map = cv2.bitwise_not(var_normalized)
    uncertainty_map = std_normalized

    return uncertainty_map

# Generate uncertainty maps
start = 0
end = 33
count = 150 # change
not_uncer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,23]  # change
data_folder = "../data_sm/stereomisp1/"
uncer_map_folder = os.path.join(data_folder, 'uncertainty_maps')
os.makedirs(uncer_map_folder, exist_ok=True)
for j in tqdm(range(count)): 
    image_paths = [os.path.join(data_folder, f"depth_{i}/{j:06d}.png") for i in range(start, end+1) if i not in not_uncer]
    uncertainty_map = generate_uncertainty_map(image_paths)
    cv2.imwrite(os.path.join(uncer_map_folder,f'{j:06d}.png'), uncertainty_map)

# form a video
images = []
for i in range(count):
    image = cv2.imread(os.path.join(uncer_map_folder, f"{i:06d}.png"))
    images.append(image)

imageio.mimwrite(os.path.join(data_folder,'videos','uncertainty_maps.mp4'), images, fps=15)