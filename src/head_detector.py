import cv2
import numpy as np
import torch
from lsc_cnn.model import LSCCNN
from lsc_cnn.utils_model import get_upsample_output, get_box_and_dot_maps

class HeadDetector(object):
    def __init__(self, checkpoint_path):
        self.network = LSCCNN(checkpoint_path=checkpoint_path)
        self.network.cuda()
        self.network.eval();
        self.nms_thresh = 0.25

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Params
        prediction_downscale = self.network.output_downscale

        # from predict_single_image
        if image.shape[0] % 16 or image.shape[1] % 16:
            image = cv2.resize(image, (image.shape[1]//16*16, image.shape[0]//16*16))
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.network.forward(img_tensor.cuda())
        out = get_upsample_output(out, prediction_downscale)
        pred_dot_map, pred_box_map = get_box_and_dot_maps(out, self.nms_thresh, self.network.BOXES)

        h_map = pred_box_map
        w_map = pred_box_map
        gt_pred_map = pred_dot_map

        # from get_boxed_img
        if image.shape[2] != 3:
            boxed_img = image.astype(np.uint8).transpose((1, 2, 0)).copy()
        else:
            boxed_img = image.astype(np.uint8).copy()
        head_idx = np.where(gt_pred_map > 0)

        H, W = boxed_img.shape[:2]
        Y, X = head_idx[-2] , head_idx[-1]
        dets = []

        for y, x in zip(Y, X):
            h, w = h_map[y, x]*prediction_downscale, w_map[y, x]*prediction_downscale        
            x1 = max(int(prediction_downscale * x - w / 2), 0)
            y1 = max(int(prediction_downscale * y - h / 2), 0)
            x2 = min(int(prediction_downscale * x + w - w / 2), W)
            y2 = min(int(prediction_downscale * y + h - h / 2), H)

            dets.append([x1, y1, w, h])

            cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0))

        cv2.imwrite("boxed_img.jpg", boxed_img)

        return dets


if __name__ == '__main__':
    img = cv2.imread("/mnt/disk/Dataset/tracking/MOT20/test/MOT20-08/img1/000001.jpg")
    extr = HeadDetector('../../lsc-cnn/weights/qnrf_scale_4_epoch_46_weights.pth')
    dets = extr(img)
