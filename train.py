# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json
from detectron2.utils.visualizer import ColorMode

from MyTrainer import MyTrainer

dataset_dir = os.path.join('land_segmentation_coco_dataset')
train_dataset_dir = os.path.join(dataset_dir, 'train')
train_images_dir = os.path.join(train_dataset_dir, 'images')
train_annotations_dir = os.path.join(train_dataset_dir, 'land_segmentation_coco.json')
val_dataset_dir = os.path.join(dataset_dir, 'val')
val_images_dir = os.path.join(val_dataset_dir, 'images')
val_annotations_dir = os.path.join(val_dataset_dir, 'land_segmentation_coco.json')

register_coco_instances(name="land_segmentation_train", metadata={}, json_file=train_annotations_dir, image_root=train_images_dir)
register_coco_instances(name="land_segmentation_val", metadata={}, json_file=val_annotations_dir, image_root=val_images_dir)
land_segmentation_metadata = MetadataCatalog.get("land_segmentation_train")
land_segmentation_metadata.thing_classes = ['Tarımsal niteliği korunacak alan','Gelişme Konut Alanı','Yerleşik Konut Alanı','Park']
land_segmentation_metadata.thing_colors = [(245,147,49),(178,80,80),(52,209,183),(61,245,61)]


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("land_segmentation_train",)
cfg.DATASETS.TEST = ("land_segmentation_val",)
cfg.TEST.EVAL_PERIOD = 200
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
#cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 39375    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
print('LR DECAY:')
print('\tSteps:', cfg.SOLVER.STEPS)
print('\tGamma:', cfg.SOLVER.GAMMA)
#cfg.SOLVER.STEPS = []        # do not decay learning rate
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

config_save_path = os.path.join(cfg.OUTPUT_DIR, 'config.yaml')
with open(config_save_path, "w") as f: 
    f.write(cfg.dump())
    
print('Train process is finished')