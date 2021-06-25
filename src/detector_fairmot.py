import torch
#import torchvision.transforms as transforms
import numpy as np
import cv2
import torch.nn.functional as F

from fairmot.opts_fairmot import opts
from fairmot.model.model import create_model, load_model
from fairmot.model.decode import mot_decode
from fairmot.model.utils import _tranpose_and_gather_feat
from fairmot.utils.post_process import ctdet_post_process

def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

class FairMOT(object):
    def __init__(self):
        self.opt = opts().init()
        if self.opt.gpus[0] >= 0:
            self.opt.device = torch.device('cuda')
        else:
            self.opt.device = torch.device('cpu')
        self.model = create_model(self.opt.arch, self.opt.heads, self.opt.head_conv)
        self.model = load_model(self.model, self.opt.load_model)
        self.model = self.model.to(self.opt.device)
        self.model.eval()

        self.width = 0
        self.height = 0

        self.max_per_image = self.opt.K

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def __call__(self, img0):
        img_size=(1088, 608)
        self.width = img_size[0]
        self.height = img_size[1]

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        im_blob = torch.from_numpy(img).cuda().unsqueeze(0)

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]
        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        dets = [[det[0], det[1], det[2]-det[0], det[3]-det[1]] for det in dets] # ltrb to xywh
        id_feature = id_feature[remain_inds]

        return dets, id_feature


if __name__ == '__main__':
    img = cv2.imread("000001.jpg")
    extr = ExtractorDet()
    dets, id_feature = extr(img)
    print(id_feature.shape, dets)
