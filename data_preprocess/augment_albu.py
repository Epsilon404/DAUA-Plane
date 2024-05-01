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


augment = [
        Blur(blur_limit=300, p=1), # 16 
        ToSepia(p=1), # 17
        ChannelDropout(channel_drop_range=(1, 1), fill_value=128, p=1), # 18
        MultiplicativeNoise(multiplier=[0.9, 1.1], elementwise=True, p=1), # 19
        MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1), # 20
        ChannelShuffle(p=1), # 21
        ToGray(p=1), # 22
        InvertImg(p=1), # 23
        RandomGamma(gamma_limit=(80, 120), p=1), # 24
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1), # 25
        RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1), # 26
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1), # 27
        MotionBlur(blur_limit=7, p=1), # 28 may have a little motion on images
        MedianBlur(blur_limit=7, p=1), # 29
        GaussianBlur(blur_limit=7, p=1), # 30
        GaussNoise(var_limit=(50.0, 100.0), p=1), # 31
        CLAHE(clip_limit=4.0, p=1), # 32
        ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1), # 33



    ] # change

start = 16 # change

data_folder = "../data_sm/stereomisp1/"
left_folder = data_folder+"images"
right_folder = data_folder+"images_right"
os.makedirs(os.path.join(data_folder,'videos'),exist_ok=True)

for n in range(len(augment)):
    num = n+start
    transform = augment[n]

    target_left = data_folder+f"images_{num}"
    target_right = data_folder+f"images_right_{num}"
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
    imageio.mimwrite(os.path.join(data_folder,'videos',f'aug_{num}.mp4'), images, fps=15)
