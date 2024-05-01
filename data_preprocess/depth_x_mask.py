import os,shutil,imageio
from tqdm import tqdm
import numpy as np

data_folder = "../data/pulling_0411_DAM/"
source_depth_folder = data_folder+"depth_0"
source_mask_folder = "../data/pulling_gt_masks"
target = "../data/pulling_masked_DAM"
os.makedirs(target,exist_ok=True)

depth_images = [f for f in os.listdir(source_depth_folder) if f.endswith(".png")]
mask_images = [f for f in os.listdir(source_mask_folder) if f.endswith(".png")]
depth_images.sort()
mask_images.sort()

for i in tqdm(range(len(depth_images))):
    source_depth = os.path.join(source_depth_folder, depth_images[i])
    source_mask = os.path.join(source_mask_folder, mask_images[i])

    mask_invert = 1-(imageio.v2.imread(source_mask,mode='L')/255)
    depth_x_mask = (imageio.v2.imread(source_depth,mode='L') * mask_invert.astype(np.uint8)).astype(np.uint8)

    target_image = os.path.join(target, f"{i:06d}.png")
    imageio.imwrite(target_image, depth_x_mask)
