import os
import json

from copy import deepcopy
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from typing import Tuple


def augment_particles(input_directory: str, output_directory: str, size: Tuple[int, int] = (1000, 1000)):
    os.makedirs(output_directory, exist_ok=True)
    json_files = [file for file in os.listdir(input_directory) if file.endswith('.json')]
    for json_filename in json_files:
        with open(os.path.join(input_directory, json_filename), 'r') as f:
            data = json.load(f)

        im: Image = Image.open(BytesIO(base64.b64decode(data['imageData'])))
        im_resized: Image = resizeImage(data, im, size)
        assert data['imageWidth'] == size[0]
        assert data['imageHeight'] == size[1]
        im_resized.save(os.path.join(output_directory, data['imagePath']))
        with open(os.path.join(output_directory, json_filename), 'w') as f:
            json.dump(data, f)

        create_and_save_patchImages(data, im_resized, json_filename, output_directory, size)
        save_rotated_and_patched_images(data, im_resized, json_filename, output_directory, size)

        print('completed', json_filename)


def resizeImage(data, im, size) -> Image:
    im_resized = im.resize(size, Image.ANTIALIAS)
    # Change imageHeight and imageWidth in json
    data['imageWidth'] = size[0]
    data['imageHeight'] = size[1]
    # Change imageData
    buffered = BytesIO()
    im_resized.save(buffered, format="JPEG")
    data['imageData'] = base64.b64encode(buffered.getvalue()).decode()
    # Change datapoints
    width_ratio = im_resized.size[0] / im.size[0]
    height_ratio = im_resized.size[1] / im.size[1]
    for annotation in data['shapes']:
        resized_points = []
        for point in annotation['points']:
            resized_points.append([point[0] * width_ratio, point[1] * height_ratio])
        annotation['points'] = resized_points
    return im_resized


def save_rotated_and_patched_images(data: dict, im_resized: Image, json_filename: str, output_directory: str, size: Tuple[int, int]) -> None:
    center = [size[0] / 2, size[1] / 2]
    imgName = os.path.basename(data['imagePath']).split('.')[0]
    for i, rotCode in enumerate([Image.ROTATE_270, Image.ROTATE_180, Image.ROTATE_90]):
        cosTheta, sinTheta = np.cos(np.pi / 2), np.sin(np.pi / 2)
        buffered = BytesIO()
        im_rotated = im_resized.transpose(rotCode)
        im_rotated.save(buffered, format="JPEG")
        data['imageData'] = base64.b64encode(buffered.getvalue()).decode()
        for annotation in data['shapes']:
            rotatet_points = []
            for point in annotation['points']:
                xCentered, yCentered = point[0] - center[0], point[1] - center[1]
                xRot, yRot = xCentered * cosTheta - yCentered * sinTheta, yCentered * cosTheta + xCentered * sinTheta
                rotatet_points.append([xRot + center[0], yRot + center[1]])
            annotation['points'] = rotatet_points

        newName = f'{imgName}_rotated_{i + 1}.jpg'
        imgPath = os.path.join(output_directory, newName)
        im_rotated.save(imgPath)

        data['imagePath'] = newName
        json_name = json_filename.split('.')[0]
        new_jsonFileName = f'{json_name}_rotated_{i + 1}.json'
        with open(os.path.join(output_directory, new_jsonFileName), 'w') as f:
            json.dump(data, f)
            
        create_and_save_patchImages(data, im_rotated, new_jsonFileName, output_directory, size)


def create_and_save_patchImages(data_orig: dict, image: Image, json_filename: str, output_directory: str, size: Tuple[int, int]) -> None:
    """
    Scale down the particles and place a small version on it (as a "patch") on an empty image. To simulate
    patches of smaller particles on the filter, these are otherwise poorly recognized...
    :param data_orig:
    :param image:
    :param json_filename:
    :param output_directory:
    :param size:
    :return:
    """
    factor = 1/5
    newSize: Tuple[int, int] = int(round(size[0]*factor)), int(round(size[1]*factor))
    emptyImg: np.ndarray = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    im_patch: np.ndarray = np.array(image.resize(newSize, Image.ANTIALIAS))
    maxX0, maxY0 = size[0] - newSize[0], size[1] - newSize[1]
    imgName = os.path.basename(data_orig['imagePath']).split('.')[0]
    for i in range(2):
        data = deepcopy(data_orig)
        x0, y0 = int(round(np.random.rand() * maxX0)), int(round(np.random.rand() * maxY0))

        newImg = emptyImg.copy()
        newImg[y0:y0+newSize[1], x0:x0+newSize[0]] = im_patch
        newImg: Image = Image.fromarray(newImg)

        buffered = BytesIO()
        newImg.save(buffered, format="JPEG")
        data['imageData'] = base64.b64encode(buffered.getvalue()).decode()

        for annotation in data['shapes']:
            resized_points = []
            for point in annotation['points']:
                resized_points.append([point[0]*factor + x0, point[1]*factor + y0])
            annotation['points'] = resized_points

        newImgName = f"{imgName}_patched_{i+1}.jpg"
        data['imagePath'] = newImgName
        newImg.save(os.path.join(output_directory, newImgName))
        json_name = json_filename.split('.')[0]
        new_jsonFileName = f'{json_name}_patched_{i+1}.json'
        with open(os.path.join(output_directory, new_jsonFileName), 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    input_dir: str = r'C:\Users\xbrjos\Desktop\particleAnnotations\annotated'
    output_dir: str = r'C:\Users\xbrjos\Desktop\particleAnnotations\augmented'
    augment_particles(input_dir, output_dir)
