import os
import json
from copy import deepcopy
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from typing import Tuple, List
from dataclasses import dataclass
import random
import cv2


@dataclass
class Bbox:
    x0: int
    y0: int
    width: int
    height: int


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

    def get_shapes(self) -> List[dict]:
        return self.data["shapes"]

    def add_shape_dict(self, newShapeDict: dict) -> None:
        self.data["shapes"].append(newShapeDict)

    def cleanup_shapes(self) -> None:
        """
        Cleans up the defined shapes, such that non of the shapes overlaps any other.
        :return:
        """
        shapeList: List[dict] = self.get_shapes()
        shapesToCheck: List[dict] = deepcopy(shapeList)
        imgShape: Tuple[int, int] = self.data['imageHeight'], self.data['imageWidth']
        cleanShapes: List[dict] = []
        for shapeDict in shapeList:
            shapeMask: np.ndarray = get_shape_mask(shapeDict["points"], imgShape)
            shapesToCheck.remove(shapeDict)
            othersMask: np.ndarray = get_mask_of_shapes(shapesToCheck, imgShape)
            shapeMask[othersMask > 0] = 0
            cleanShapes += get_shape_dicts_from_mask(shapeMask, shapeDict)

        self.data["shapes"] = cleanShapes


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


def create_copy_paste_images(img1: ImageDefinition, img2: ImageDefinition, numVariations: int = 3) -> List[ImageDefinition]:
    """
    Will create copy-pasta-variations (numVariations)
    :param img1: The image to paste particles into
    :param img2: The image to copy particles from
    :param numVariations: number of variations to produce
    :return: List of new ImageDefinition Objects
    """
    random.seed(42)
    np.random.seed(42)
    newImages: List[ImageDefinition] = []

    for i in range(numVariations):
        sourceImg, targetImg = img1.getDeepCopy(), img2.getDeepCopy()
        sourceImgArr: np.ndarray = np.array(sourceImg.imgObj)
        targetImgArr: np.ndarray = np.array(targetImg.imgObj)

        sourceParticles: List[dict] = sourceImg.get_shapes()
        numParticlesToCopy: int = int(round(np.random.rand()*len(sourceParticles)/2))

        for particle in random.sample(sourceParticles, numParticlesToCopy):
            mask, particleImg, newParticleDict = get_particle_img_with_random_offset(particle, sourceImgArr)
            targetImgArr[mask > 0] = particleImg[mask > 0]
            targetImg.add_shape_dict(newParticleDict)

        targetImg.imgObj = Image.fromarray(targetImgArr)
        targetImg.cleanup_shapes()
        targetImg.add_suffix(f"_copyPaste {i+1}")
        newImages.append(targetImg)

    return newImages


def get_mask_of_shapes(shapeList: List[dict], imgShape: Tuple[int, int]) -> np.ndarray:
    othersMask: np.ndarray = np.zeros(imgShape, dtype=int)
    for otherShapeDict in shapeList:
        othersMask += get_shape_mask(otherShapeDict["points"], imgShape)
    return othersMask


def get_particle_img_with_random_offset(particleDict: dict, sourceImage: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns an image of the particle from the sourceImage. A random offset is applied to the particle image and the mask.
    Everything except the pixels within the particle shape is black.
    :param particleDict: Particle (shape) dictionary
    :param sourceImage: Source image to use
    :return: Tuple[new mask, particle Image, new particle Dict]
    """
    img: np.ndarray = np.zeros_like(sourceImage)

    cnt: np.ndarray = get_shape_contour(particleDict["points"])
    mask: np.ndarray = np.zeros(sourceImage.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 1, -1)
    img[mask > 0] = sourceImage[mask > 0]

    bbox: Bbox = get_contour_bbox(cnt)
    newy0: int = int(round(np.random.rand() * (img.shape[0] - bbox.height)))
    newx0: int = int(round(np.random.rand() * (img.shape[1] - bbox.width)))

    newMask: np.ndarray = np.zeros_like(mask)
    particleMask: np.ndarray = mask[bbox.y0:bbox.y0 + bbox.height, bbox.x0:bbox.x0 + bbox.width]
    newMask[newy0:newy0+bbox.height, newx0:newx0+bbox.width] = particleMask

    newParticleImg: np.ndarray = np.zeros_like(sourceImage)
    particleImg: np.ndarray = img[bbox.y0:bbox.y0 + bbox.height, bbox.x0:bbox.x0 + bbox.width, :]
    newParticleImg[newy0:newy0 + bbox.height, newx0:newx0 + bbox.width, :] = particleImg

    newParticleDict: dict = deepcopy(particleDict)
    diffx, diffy = newx0 - bbox.x0, newy0 - bbox.y0
    for point in newParticleDict["points"]:
        point[0] = int(round(point[0] + diffx))
        point[1] = int(round(point[1] + diffy))

    return newMask, newParticleImg, newParticleDict


def get_contour_bbox(cnt: np.ndarray) -> Bbox:
    """
    Computes the bounding box extrema of the given contour
    :param cnt: np.ndarray of contour
    :return: y0, y1, x0, x1
    """
    y0, y1 = cnt[:, 0, 1].min(), cnt[:, 0, 1].max()
    x0, x1 = cnt[:, 0, 0].min(), cnt[:, 0, 0].max()
    return Bbox(x0, y0, x1-x0, y1-y0)


def get_shape_mask(shapePoints: List[List[int]], imgShape: Tuple[int, int]) -> np.ndarray:
    mask: np.ndarray = np.zeros(imgShape, dtype=np.uint8)
    cnt: np.ndarray = get_shape_contour(shapePoints)
    cv2.drawContours(mask, [cnt], -1, 1, -1)
    return mask


def get_shape_contour(shapePoints: List[List[float]]) -> np.ndarray:
    return np.expand_dims(np.array(shapePoints), 1).astype(np.int)


def get_shape_dicts_from_mask(mask: np.ndarray, shapeTemplate: dict) -> List[dict]:
    shapes: List[dict] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    for cnt in contours:
        newShape: dict = deepcopy(shapeTemplate)
        newShape["points"] = get_points_from_cnt(cnt)
        shapes.append(newShape)
    return shapes


def get_points_from_cnt(contour: np.ndarray) -> List[List[float]]:
    return [[float(p[0][0]), float(p[0][1])] for p in contour]
