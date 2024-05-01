import os,shutil,imageio
from tqdm import tqdm
import numpy as np

data_folder = "../data_cutting/cutting_0411_DAM/"
source_uncer_folder = data_folder+"uncertainty_maps"
source_mask_folder = "../data_cutting/cutting_gt_masks"
target = "../data_cutting/cutting_masked_uncer0411"
os.makedirs(target,exist_ok=True)

uncer_images = [f for f in os.listdir(source_uncer_folder) if f.endswith(".png")]
mask_images = [f for f in os.listdir(source_mask_folder) if f.endswith(".png")]
uncer_images.sort()
mask_images.sort()

for i in tqdm(range(len(uncer_images))):
    source_uncer = os.path.join(source_uncer_folder, uncer_images[i])
    source_mask = os.path.join(source_mask_folder, mask_images[i])

    mask_invert = 1-(imageio.v2.imread(source_mask,mode='L')/255)
    uncer_x_mask = (imageio.v2.imread(source_uncer,mode='L') * mask_invert.astype(np.uint8)).astype(np.uint8)

    target_image = os.path.join(target, f"{i:06d}.png")
    imageio.imwrite(target_image, uncer_x_mask)

# form a video
images = []
for i in range(len(uncer_images)):
    image = imageio.v2.imread(os.path.join(target, f"{i:06d}.png"))
    images.append(image)

imageio.mimwrite(os.path.join(data_folder,'videos','uncertainty_x_mask.mp4'), images, fps=15)