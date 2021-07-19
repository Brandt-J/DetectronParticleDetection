import random

import torch
assert torch.__version__.startswith("1.8")
gpuAvailable = torch.cuda.is_available()
print('Cuda Availabe:', gpuAvailable)

from detectron2.utils.logger import setup_logger
setup_logger()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer

from functions import *

if __name__ == '__main__':
    allImgsPath = r'C:\Users\xbrjos\Desktop\particleAnnotations\augmented'
    otherImgsPath = r"C:\Users\xbrjos\Desktop\particleAnnotations\others"  # Images NOT in test set

    trainGetter, testGetter = get_trainTest_getters(allImgsPath, 0.1)
    for subset, getter in zip(["train", "test"], [trainGetter, testGetter]):
        DatasetCatalog.register("particles_" + subset, getter)
        MetadataCatalog.get("particles_" + subset).set(thing_classes=['Particle', 'Fibre'])

    MetadataCatalog.get("particles_test").set(evaluator_type="sem_seg")
    particles_metadata = MetadataCatalog.get("particles_train")

    # Training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("particles_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 50
    cfg.SOLVER.CHECKPOINT_PERIOD = 5
    cfg.TEST.EVAL_PERIOD = 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    if not gpuAvailable:
        cfg.MODEL.DEVICE = 'cpu'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # Inference
    weihtsPath = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if not os.path.exists(weihtsPath):
        print('cannot find weights at:', weihtsPath, ', not running model inference.')
    else:
        cfg.MODEL.WEIGHTS = weihtsPath
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # was 0.5
        cfg.DATASETS.TEST = ("particles_test", )
        predictor = DefaultPredictor(cfg)

        outImgFolder = 'EvaluationImages'
        os.makedirs(outImgFolder, exist_ok=True)
        dataset_dicts = DatasetCatalog.data["particles_test"]()
        for i, d in enumerate(random.sample(dataset_dicts, 5)):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=particles_metadata,
                           scale=1.0,
                           instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(figsize=(14, 10))
            plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            plt.savefig(os.path.join(outImgFolder, f"test image {i+1}.png"))

        for imgPath in random.sample(os.listdir(otherImgsPath), 5):
            im = cv2.imread(os.path.join(otherImgsPath, imgPath))
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=particles_metadata,
                           scale=1.0,
                           instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(figsize=(14, 10))
            plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            plt.savefig(os.path.join(outImgFolder, f"test image {os.path.basename(imgPath).split('.')[0]}.png"))
