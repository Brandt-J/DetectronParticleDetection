import numpy as np
import json
import os
from typing import Tuple, List, TYPE_CHECKING
from detectron2.structures import BoxMode


def get_trainTest_getters(directory: str, fracTest: float = 0.1) -> Tuple['function', 'function']:
    # numSamples = get_numDsets_in_dir(directory)
    allDsets: List[dict] = get_particle_dicts(directory)
    numSamples = len(allDsets)
    numTest = round(numSamples * fracTest)
    numTrain = numSamples - numTest
    return lambda: get_list_slice(allDsets, 0, numTrain), lambda: get_list_slice(allDsets, numTrain, numSamples)


def get_particle_dicts(directory) -> List[dict]:
    classes = ['Particle', 'Fibre']
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}

        filename = os.path.join(directory, img_anns["imagePath"])

        record["file_name"] = filename
        record["height"] = 1000
        record["width"] = 1000

        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            if len(px) > 4:
                poly = [(x, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(anno['label']),
                    "iscrowd": 0
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_numDsets_in_dir(directory: str) -> int:
    return len([file for file in os.listdir(directory) if file.endswith('.json')])


def get_list_slice(listItem: list, startInd: int, endInd: int) -> list:
    return listItem[startInd:endInd]