import os
import json
import random

from PIL import Image
from io import BytesIO
import base64
from typing import Tuple, List
from itertools import combinations

from augMethods import ImageDefinition, resizeImage, create_rotated_images, create_patch_images, create_lowRes_version, create_copy_paste_images


def augment_particles(input_directory: str, output_directory: str, size: Tuple[int, int] = (1000, 1000)):
    os.makedirs(output_directory, exist_ok=True)
    images: List[ImageDefinition] = get_images_from_directory(input_directory)
    print(f'Starting with {len(images)} original images.')
    images = list(map(lambda x: resizeImage(x, size), images))
    imageCombinations: List[Tuple[ImageDefinition, ...]] = list(combinations(images, 2))

    for img1, img2 in random.sample(imageCombinations, int(round(len(imageCombinations)/3))):  # only take 1/3 of all possible combinations
        images += create_copy_paste_images(img1, img2, numVariations=2)

    print(f"Having {len(images)} images after copy-paste augmentation.")

    for i, img in enumerate(images):
        resized: ImageDefinition = resizeImage(img, size)
        lowRes: Image = create_lowRes_version(resized)

        for edited in [resized, lowRes]:
            edited.saveImage(output_directory)
            for rotated in create_rotated_images(edited):
                rotated.saveImage(output_directory)
                for patched in create_patch_images(rotated, numVersions=1):
                    patched.saveImage(output_directory)

        print(f'completed augmentation of {img.imgName}, {round((i+1)/len(images) * 100)} % done')
    print(f"Saved in total {ImageDefinition.get_save_counter()} images.")


def get_images_from_directory(directory: str) -> List[ImageDefinition]:
    imgs: List[ImageDefinition] = []
    json_files = [file for file in os.listdir(directory) if file.endswith('.json')]
    for i, json_filename in enumerate(json_files):
        with open(os.path.join(directory, json_filename), 'r') as f:
            data = json.load(f)

        imgName: str = json_filename.split('.')[0]
        im: Image = Image.open(BytesIO(base64.b64decode(data['imageData'])))
        newImg: ImageDefinition = ImageDefinition(imgName, data, im)
        imgs.append(newImg)
    return imgs


if __name__ == '__main__':
    input_dir: str = r'C:\Users\xbrjos\Desktop\particleAnnotations\annotated'
    output_dir: str = r'C:\Users\xbrjos\Desktop\particleAnnotations\augmented'
    augment_particles(input_dir, output_dir)
