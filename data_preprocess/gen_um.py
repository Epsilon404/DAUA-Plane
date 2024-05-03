import torchvision.transforms as transforms
import os
from PIL import Image
from tqdm import tqdm
from albumentations import (
    Blur, RandomGamma, HueSaturationValue, RGBShift, RandomBrightnessContrast, 
    MotionBlur, MedianBlur, GaussianBlur, GaussNoise, CLAHE, InvertImg, ChannelShuffle, ISONoise, 
    OneOf, Compose,

    MultiplicativeNoise, ToSepia, ChannelDropout, ToGray
)
import numpy as np

need_same = True

# ------------- Augmentation -------------
augment = [
        Blur(blur_limit=300, p=1),
        ToSepia(p=1), 
        ChannelDropout(channel_drop_range=(1, 1), fill_value=128, p=1), 
        MultiplicativeNoise(multiplier=[0.9, 1.1], elementwise=True, p=1), 
        MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1), 
        ChannelShuffle(p=1), 
        ToGray(p=1), 
        InvertImg(p=1), 
        RandomGamma(gamma_limit=(80, 120), p=1), 
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1), 
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1), 
        MotionBlur(blur_limit=7, p=1), 
        MedianBlur(blur_limit=7, p=1), 
        GaussianBlur(blur_limit=7, p=1),
        GaussNoise(var_limit=(50.0, 100.0), p=1),
        CLAHE(clip_limit=4.0, p=1), 
        ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1), 



    ]

start = 1

data_name = "cutting_tissue_twice" # change
# data_name = "pulling_soft_tissues" # change

data_folder = "../data/endonerf_full_datasets/"+data_name+"/"
left_folder = data_folder+"images"
right_folder = data_folder+"images_right"
output_folder = "./processing/"+data_name+"_tmp/"
image_count = len([f for f in os.listdir(left_folder) if f.endswith(".png")])
os.makedirs(os.path.join(output_folder,'videos'),exist_ok=True)

for n in range(len(augment)):
    num = n+start
    transform = augment[n]

    target_left = output_folder+f"images_{num}"
    target_right = output_folder+f"images_right_{num}"
    os.makedirs(target_left,exist_ok=True)
    os.makedirs(target_right,exist_ok=True)

    left_images = [f for f in os.listdir(left_folder) if f.endswith(".png")]
    right_images = [f for f in os.listdir(right_folder) if f.endswith(".png")]
    left_images.sort()
    right_images.sort()

    # Apply the transformation to each image
    aug_left_images = []
    aug_right_images = []
    for i,(left,right) in enumerate(tqdm(zip(left_images,right_images),desc=f"Augmenting {num}")):
        left = np.array(Image.open(os.path.join(left_folder, left)))
        right = np.array(Image.open(os.path.join(right_folder, right)))

        if need_same:
            two_img = np.hstack((left,right))
            aug_two_img = transform(image=two_img)['image']
            augmented_left = aug_two_img[:, :left.shape[1]]
            augmented_right = aug_two_img[:, left.shape[1]:]
        else:
            augmented_left = transform(image=left)['image']
            augmented_right = transform(image=right)['image']
        
        target_left_image = os.path.join(target_left, f"{i:06d}.png")
        target_right_image = os.path.join(target_right, f"{i:06d}.png")
        Image.fromarray(augmented_left).save(target_left_image)
        Image.fromarray(augmented_right).save(target_right_image)

        aug_left_images.append(target_left_image)
        aug_right_images.append(target_right_image)


    # form a video
    import imageio
    import numpy as np

    images = []
    for i,(l,r) in enumerate(zip(aug_left_images,aug_right_images)):
        left = np.array(Image.open(l))
        right = np.array(Image.open(r))
        images.append(np.hstack((left,right)))
    imageio.mimwrite(os.path.join(output_folder,'videos',f'aug_{num}.mp4'), images, fps=15)


# ------------- DAM -------------
from transformers import pipeline
import imageio

# load pipe
pipe = pipeline(task="depth-estimation", model="./depth-anything-large-hf")


start = 0
end = len(augment)

for num in range(start,end+1):

    if num == 0:
        source_folder = data_folder+"images"
    else:
        source_folder = output_folder+f"images_{num}"

    target_folder = output_folder+f"depth_{num}"
    os.makedirs(target_folder,exist_ok=True)

    image_files = [f for f in os.listdir(source_folder) if f.endswith(".png")]
    image_files.sort()

    for i, image_file in enumerate(tqdm(image_files,desc=f"Generating depth_{num}")):
        source_image = os.path.join(source_folder, image_file)
        target_image = os.path.join(target_folder, f"{i:06d}.png")

        image = Image.open(source_image)
        depth = pipe(image)["depth"]

        imageio.imwrite(target_image, depth)
        d_invert = (255-imageio.v2.imread(target_image, mode='L'))/2-10
        d_invert = np.maximum(d_invert,0)
        d_invert = d_invert.astype(np.uint8)
        imageio.imwrite(target_image,d_invert)

    # form a video
    images = []
    for i in tqdm(range(len(image_files))):
        depth = np.array(Image.open(os.path.join(target_folder, f"{i:06d}.png")))
        images.append(depth)

    imageio.mimwrite(os.path.join(output_folder,'videos',f'depth_{num}.mp4'), images, fps=15)



# ------------- uncertainty map -------------
import cv2

def generate_uncertainty_map(image_paths):
    # Load the images
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    std = np.std(images, axis=0)
    std_normalized = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX)
    std_normalized = std_normalized.astype(np.uint8)
    uncertainty_map = std_normalized

    return uncertainty_map

# Generate uncertainty maps

count = image_count
uncer_map_folder = output_folder+"uncer_map"
os.makedirs(uncer_map_folder)
for j in tqdm(range(count)): 
    image_paths = [os.path.join(output_folder, f"depth_{i}/{j:06d}.png") for i in range(start, end+1)]
    uncertainty_map = generate_uncertainty_map(image_paths)
    cv2.imwrite(os.path.join(uncer_map_folder,f'{j:06d}.png'), uncertainty_map)

# form a video
images = []
for i in range(count):
    image = cv2.imread(os.path.join(uncer_map_folder, f"{i:06d}.png"))
    images.append(image)

imageio.mimwrite(os.path.join(output_folder,'videos','uncertainty_maps.mp4'), images, fps=15)



# ------------- uncer map x mask  -------------

source_uncer_folder = uncer_map_folder
source_mask_folder = data_folder+"gt_masks"
target = data_folder+"uncer_map"
os.makedirs(target)

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

imageio.mimwrite(os.path.join(output_folder,'videos','uncertainty_x_mask.mp4'), images, fps=15)



# ------------- depth x mask -------------

source_depth_folder = output_folder+"depth_0"
source_mask_folder = data_folder+"gt_masks"
target = data_folder+"depth_DAM"
os.makedirs(target)

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
