from transformers import pipeline
from PIL import Image
import requests
import imageio
from tqdm import tqdm
import os
import numpy as np

# load pipe
pipe = pipeline(task="depth-estimation", model="./depth-anything-large-hf")

# load image
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

data_folder = "../data_sm/stereomisp1/"
start = 0  # change
end = 33  # change
not_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # change

for num in range(start,end+1):

    if num == 0:
        source_folder = data_folder+"images_right"
    elif num in not_depth:
        continue
    else:
        source_folder = data_folder+f"images_right_{num}"

    target_folder = data_folder+f"depth_{num}"
    os.makedirs(target_folder,exist_ok=True)

    image_files = [f for f in os.listdir(source_folder) if f.endswith(".png")]
    image_files.sort()

    for i, image_file in enumerate(tqdm(image_files,desc=f"Generating depth_{num}")):
        source_image = os.path.join(source_folder, image_file)
        target_image = os.path.join(target_folder, f"{i:06d}.png")

        image = Image.open(source_image)
        depth = pipe(image)["depth"]

        imageio.imwrite(target_image, depth)
        # d_invert = (255-imageio.v2.imread(target_image, mode='L'))/2-10 # cutting
        # d_invert = (255-imageio.v2.imread(target_image, mode='L'))/2-30 # pulling
        d_invert = (255-imageio.v2.imread(target_image, mode='L'))/4.5+30 # stereomis
        d_invert = np.maximum(d_invert,0)
        d_invert = d_invert.astype(np.uint8)
        imageio.imwrite(target_image,d_invert)

    # form a video
    images = []
    for i in tqdm(range(len(image_files))):
        depth = np.array(Image.open(os.path.join(target_folder, f"{i:06d}.png")))
        images.append(depth)

    imageio.mimwrite(os.path.join(data_folder,'videos',f'depth_{num}.mp4'), images, fps=15)


    
# source_image = os.path.join(source_folder, image_files[0])
# target_image = os.path.join(target_folder, "0.png")
# cutting_depth = imageio.v2.imread("./cutting_tissues_twice/gt_depth/000000.png",mode='L')
# image = Image.open(source_image)
# depth = pipe(image)["depth"]
# imageio.imwrite(target_image, depth)

# d = (255-imageio.v2.imread(target_image, mode='L'))  # / 3.6 + 20
# d_median = np.median(d)
# cutting_median = np.median(cutting_depth)
# # print(np.median(d),np.median(cutting_depth))
# d = d * cutting_median / d_median
# d = d.astype(np.uint8)
# imageio.imwrite(target_image, d)