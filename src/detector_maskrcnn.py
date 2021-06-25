import cv2
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


def setup_cfg(config_path, option_list, thre_conf):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(option_list)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thre_conf
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thre_conf
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = thre_conf
    cfg.freeze()

    return cfg

class MaskRCNN(object):
    def __init__(self, config_path, option_list, thre_conf):
        cfg = setup_cfg(config_path, option_list, thre_conf)
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, image):
        predictions = self.predictor(image)
        instances = predictions["instances"]

        # Classes
        classes = instances.pred_classes
        classes = classes.cpu().clone().numpy()
        classes = np.where(classes==0)[0] # Only class "0"

        # Dets
        dets = instances.pred_boxes
        dets = dets.tensor.cpu().clone().numpy()
        dets = [[det[0], det[1], det[2]-det[0], det[3]-det[1]] for det in dets]
        dets = np.array(dets)
        dets = dets[classes]

        # Scores
#        scores = instances.scores
#        scores = scores.cpu().clone().numpy()

        # Masks
        masks = instances.pred_masks
        masks = masks.cpu().clone().numpy()
        masks = masks[classes]

        return dets, masks


if __name__ == '__main__':
    img = cv2.imread("/mnt/disk/Dataset/tracking/MOT16/train/MOT16-04/img1/000001.jpg")
    extr = MaskRCNN("../detectron2_configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", ["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"], 0.5)
    dets, masks = extr(img)
    mask = masks[0]
    mask = mask.astype(np.uint8)
    mask = mask * 255
    cv2.imwrite("mask.jpg", mask)
    print(len(masks))
