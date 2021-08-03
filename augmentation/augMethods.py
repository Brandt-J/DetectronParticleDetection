import os
import json
from copy import deepcopy
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class ImageDefinition:
    imgName: str
    data: dict
    imgObj: Image.Image

    def add_suffix(self, suffix: str) -> None:
        self.imgName += suffix

    def saveImage(self, directory: str) -> None:
        self.data['imagePath'] = self.imgName
        self.imgObj.save(os.path.join(directory, self.imgName + '.png'))

        buffered = BytesIO()
        self.imgObj.save(buffered, format="JPEG")
        self.data['imageData'] = base64.b64encode(buffered.getvalue()).decode()

        new_jsonFileName = f'{self.imgName}.json'
        with open(os.path.join(directory, new_jsonFileName), 'w') as f:
            json.dump(self.data, f)

    def getDeepCopy(self) -> 'ImageDefinition':
        return ImageDefinition(deepcopy(self.imgName),
                               deepcopy(self.data),
                               deepcopy(self.imgObj))


def resizeImage(image: ImageDefinition, size: Tuple[int, int]) -> ImageDefinition:
    """
    Resize an image and return new image data and Image.
    :param image: ImageDefinition Object
    :param size: target size
    :return: Tuple[newDataDict, newImage]
    """
    newImg = image.getDeepCopy()
    data, im = newImg.data, newImg.imgObj
    im_resized = im.resize(size, Image.ANTIALIAS)
    data['imageWidth'] = size[0]
    data['imageHeight'] = size[1]

    width_ratio = im_resized.size[0] / im.size[0]
    height_ratio = im_resized.size[1] / im.size[1]
    for annotation in data['shapes']:
        resized_points = []
        for point in annotation['points']:
            resized_points.append([point[0] * width_ratio, point[1] * height_ratio])
        annotation['points'] = resized_points
    newImg.imgObj = im_resized
    newImg.data = data
    return newImg


def create_rotated_images(image: ImageDefinition) -> List[ImageDefinition]:
    """
    Create a series of three rotated images
    :param image: the ImageDefinition object
    :return: List of Tuples of (imageData, image) of rotated versions.
    """
    newImg: ImageDefinition = image.getDeepCopy()
    rotated: List[ImageDefinition] = []
    size: Tuple[int, int] = image.data['imageWidth'], image.data['imageHeight']
    center = [size[0] / 2, size[1] / 2]
    cosTheta, sinTheta = np.cos(np.pi / 2), np.sin(np.pi / 2)

    for i in range(3):
        newImg = newImg.getDeepCopy()
        data, img = newImg.data, newImg.imgObj
        im_rotated = img.transpose(Image.ROTATE_270)
        for annotation in data['shapes']:
            annotation['points'] = rotate_points(annotation, center, cosTheta, sinTheta)
        newImg.data = data
        newImg.imgObj = im_rotated
        newImg.add_suffix(f"_rotated {i+1}")

        rotated.append(newImg)
    return rotated


def rotate_points(annotation: dict, center: List[float], cosTheta: float, sinTheta: float) -> List[List[float]]:
    rotatet_points: List[List[float]] = []
    for point in annotation['points']:
        xCentered, yCentered = point[0] - center[0], point[1] - center[1]
        xRot, yRot = xCentered * cosTheta - yCentered * sinTheta, yCentered * cosTheta + xCentered * sinTheta
        rotatet_points.append([xRot + center[0], yRot + center[1]])
    return rotatet_points


def create_patch_images(image: ImageDefinition, numVersions: int = 2) -> List[ImageDefinition]:
    """
    Scale down the particles and place a small version on it (as a "patch") on an empty image. To simulate
    patches of smaller particles on the filter, these are otherwise poorly recognized...
    :param image: the ImageDefinition obect
    :param numVersions: number of versions to create
    :return List of tuples (imagedata, Image) for patched versions:
    """
    patchedImgs: List[ImageDefinition] = []
    factor = 1 / 5
    size: Tuple[int, int] = image.data['imageWidth'], image.data['imageHeight']
    newSize: Tuple[int, int] = int(round(size[0] * factor)), int(round(size[1] * factor))
    emptyImg: np.ndarray = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    im_patch: np.ndarray = np.array(image.imgObj.resize(newSize, Image.ANTIALIAS))
    maxX0, maxY0 = size[0] - newSize[0], size[1] - newSize[1]
    for i in range(numVersions):
        newImgObj: ImageDefinition = image.getDeepCopy()
        x0, y0 = int(round(np.random.rand() * maxX0)), int(round(np.random.rand() * maxY0))
        data, newImg = newImgObj.data, newImgObj.imgObj
        newImg = emptyImg.copy()
        newImg[y0:y0 + newSize[1], x0:x0 + newSize[0]] = im_patch
        newImg: Image = Image.fromarray(newImg)

        for annotation in data['shapes']:
            resized_points = []
            for point in annotation['points']:
                resized_points.append([point[0] * factor + x0, point[1] * factor + y0])
            annotation['points'] = resized_points

        newImgObj.imgObj = newImg
        newImgObj.data = data
        newImgObj.add_suffix(f"_patched {i+1}")
        patchedImgs.append(newImgObj)
    return patchedImgs


def create_lowRes_version(img: ImageDefinition, scaleFactor: float = 10.0) -> ImageDefinition:
    """
    Downscales and upsacles the image according the given scale Factor, in order to create a low_res and then
    upsampled image.
    :param img:
    :param scaleFactor:
    :return: blurred low res image
    """
    newImg: ImageDefinition = img.getDeepCopy()
    origSize: Tuple[int, int] = img.imgObj.size
    downScaledSize: Tuple[int, int] = int(round(origSize[0] / scaleFactor)), int(round(origSize[1] / scaleFactor))
    newImg.imgObj = newImg.imgObj.resize(downScaledSize)
    newImg.imgObj = newImg.imgObj.resize(origSize)
    newImg.add_suffix("_lowRes")
    return newImg
