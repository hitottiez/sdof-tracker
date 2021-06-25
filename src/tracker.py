from track import Track
from linear_assignment import *
import cv2
import sys
import pandas as pd
import math
import random

sys.path.append("../../FairMOT/src/lib")
from tracking_utils.timer import Timer

# frame_id starts from 1
def read_det(det_path, frame_id, x_max, y_max, max_size, thre_conf):
    df = pd.read_csv(det_path)
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "X", "Y", "Z"]
    df = df[df["frame"] == frame_id]
    df = df[(0 < df["x"] + df["w"]) & (df["x"] < x_max) & (0 < df["y"] + df["h"]) & (df["y"] < y_max)] # Overlap at least
    df = df[(df["w"] < max_size) & (df["h"] < max_size)]
    df = df[df["conf"] > thre_conf]
    dets = df[["x", "y", "w", "h"]].values

    return dets

# frame_id starts from 1
def read_gt(gt_path, frame_id):
    df = pd.read_csv(gt_path)
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]
    df = df[df["frame"] == frame_id]
    df = df[(df["class"] == 1) | (df["class"] == 2) | (df["class"] == 7)] # 1: pedestrian, 2: person on vehicle, 7: static person
    dets = df[["x", "y", "w", "h"]].values

    return dets

def calc_tblr(bbox, x_max, y_max):
    top = min(max(int(bbox[1]), 0), y_max-1)
    bottom = min(max(int(bbox[1] + bbox[3]), 1), y_max)
    left = min(max(int(bbox[0]), 0), x_max-1)
    right = min(max(int(bbox[0] + bbox[2]), 1), x_max)

    return top, bottom, left, right

def calc_intersection(bbox, candidate):
    print(bbox, candidate)
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidate_tl, candidate_br = candidate[:2], candidate[:2] + candidate[2:]

    tl = np.array([max(bbox_tl[0], candidate_tl[0]), max(bbox_tl[1], candidate_tl[1])])
    br = np.array([min(bbox_br[0], candidate_br[0]), min(bbox_br[1], candidate_br[1])])
    wh = np.maximum(0., br - tl)
    intersection = [tl[0], tl[1],  wh[0], wh[1]]
    print(intersection)

    return intersection

class Tracker:
    def __init__(self, det_path, gt_path, max_age, max_dist_iou, max_dist_feature, n_init, reinit_interval, metric, detect_method, point_termi, start_ind, max_size, thre_conf, thre_var_ratio, thre_homo, point_detect, focus_point_manual, focus_point_auto, use_mask,  head_detect, r_ratio, interval_num, K, max_point_num, shi_tomasi, feature_params, lk_params):
        self.start_ind = start_ind
        self.frame_ind = 1
        self.frame = None
        self.frame_old = None
        self.frame_gray_old = None
        self.frame_gray = None
        self.x_max = 0
        self.y_max = 0
        self.tracks = []
        self.next_id = 1
        self.reinit = 1 # reinit frame or not
        self.already_updated = 0 # Already updated track status by points
        self.max_age = max_age
        self.max_dist_iou = max_dist_iou
        self.max_dist_feature = max_dist_feature
        self.n_init = n_init
        self.reinit_interval = reinit_interval
        self.point_termi = point_termi
        self.thre_var_ratio = thre_var_ratio
        self.thre_homo = thre_homo
        self.thre_conf = thre_conf
        self.max_size = max_size
        self.metric = metric
        if metric == "feature":
            from feature_extractor import Extractor
            self.extractor = Extractor("../model/extractor/ckpt.t7")

        self.detect_method = detect_method
        if self.detect_method == "read_gt" or self.detect_method == "read_det":
            self.det_path = det_path
            self.gt_path = gt_path
        elif self.detect_method == "maskrcnn":
            from detector_maskrcnn import MaskRCNN
            self.detector = MaskRCNN("../detectron2_configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", ["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"], self.thre_conf)
        elif self.detect_method == "centermask":
            from detector_centermask import CenterMask
            self.detector = CenterMask("../detectron2_configs/centermask/centermask_V_99_eSE_FPN_ms_3x.yaml", ["MODEL.WEIGHTS", "../model/centermask/centermask2-V-99-eSE-FPN-ms-3x.pth"], self.thre_conf)
        elif self.detect_method == "fairmot":
            from detector_fairmot import FairMOT
            self.detector = FairMOT()

        # About point detection
        self.point_detect = point_detect
        self.focus_point_manual = focus_point_manual
        self.focus_point_auto = focus_point_auto
        self.use_mask = use_mask
        self.head_detect = head_detect
        if self.head_detect:
            from head_detector import HeadDetector
            self.head_detector = HeadDetector('../../lsc-cnn/weights/qnrf_scale_4_epoch_46_weights.pth')
        self.r_ratio = r_ratio
        self.interval_num = interval_num
        self.K = K
        self.max_point_num = max_point_num
        self.shi_tomasi = shi_tomasi
        self.feature_params = feature_params
        self.lk_params = lk_params

    def update_frame(self, frame_ind, frame):
        self.frame_ind = frame_ind
        self.frame = frame
        self.x_max = frame.shape[1]
        self.y_max = frame.shape[0]
        if frame_ind == self.start_ind:
            pass
        else:
            self.frame_old = self.frame.copy()
            self.frame_gray_old = self.frame_gray.copy()
        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_ind == 1 or self.frame_ind % self.reinit_interval == 0:
            self.reinit = 1
        else:
            self.reinit = 0

    def add_track(self, det, feature, mask):
        self.tracks.append(Track(self.next_id, det, feature, mask, self.n_init, self.max_age, self.reinit_interval))
        self.next_id += 1

    def update_status_every(self):
        for track in self.tracks:
            track.update_status_every()

    def get_all_points(self):
        points = np.array([])
        point_ids = np.array([])
        for k, track in enumerate(self.tracks):
            if len(track.points) == 0:
                continue
            elif len(points) == 0:
                points = track.points.reshape(-1, 2)
                point_ids = np.ones(len(track.points)) * k
            else:
                points = np.vstack([points, track.points])
                point_ids = np.hstack([point_ids, np.ones(len(track.points)) * k])

        return points, point_ids

    def detect_head(self):
        heads = self.head_detector(self.frame)
        matches, _, _ = linear_assignment(self.tracks, heads, [], [], 1000, self.max_age, 0)
        
        for match in matches:
            k = match[0]
            bbox = self.tracks[k].bbox
            head = heads[match[1]]

            mask = np.zeros(self.frame_gray.shape)

#            inter = calc_intersection(bbox, head)
#            cv2.rectangle(mask, (int(inter[0]), int(inter[1])), (int(inter[0] + inter[2]), int(inter[1] + inter[3])), (1), -1)

            cv2.rectangle(mask, (int(head[0]), int(head[1])), (int(head[0] + head[2]), int(head[1] + head[3])), (1), -1)
            cv2.rectangle(mask, (0, 0), (self.x_max, int(bbox[1])), (0), -1)
            cv2.rectangle(mask, (0, int(bbox[1] + bbox[3] * 0.3)), (self.x_max, self.y_max), (0), -1)
            cv2.rectangle(mask, (0, 0), (int(bbox[0]), self.y_max), (0), -1)
            cv2.rectangle(mask, (int(bbox[0] + bbox[2]), 0), (self.x_max, self.y_max), (0), -1)

            self.tracks[k].mask = mask

    def init_points(self):
        if self.head_detect:
            self.detect_head()

        if self.point_detect == "auto":
            self.init_points_auto()
        elif self.point_detect == "manual":
            self.init_points_manual()

    def init_points_auto(self):
        for k in range(len(self.tracks)):
            self.tracks[k].points = []
            bbox = self.tracks[k].bbox
            bbox = [int(a) for a in bbox]

            mask = np.zeros(self.frame_gray.shape)
            if self.use_mask:
                mask = self.tracks[k].mask
                if len(mask) == 0:
                    cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (1), -1) # all bbox
            else:
                cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (1), -1) # all bbox
            mask = mask.astype(np.uint8) 

            # Erode
            kernel = np.ones((2,2),np.uint8)
            iterations = 2
            mask = cv2.erode(mask, kernel, iterations=iterations)

            # Focus point
            if self.focus_point_auto == "head":
                cv2.rectangle(mask, (bbox[0], int(bbox[1] + bbox[3] * 0.2)), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0), -1)

            elif self.focus_point_auto == "center":
                cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[0] + bbox[2], int(bbox[1] + bbox[3] * 0.4)), (0), -1)
                cv2.rectangle(mask, (bbox[0], int(bbox[1] + bbox[3] * 0.6)), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0), -1)

            # Point detection
            points = np.array([])
            if self.shi_tomasi:
                points_auto = cv2.goodFeaturesToTrack(self.frame_gray, mask=mask, **self.feature_params)
            else:
                points_auto = None

            if points_auto is None:
                lack_num = self.max_point_num
            else:
                for point_auto in points_auto:
                    if len(points) == 0:
                        points = point_auto.reshape(-1, 2)
                    else:
                        points = np.vstack([points, point_auto])
                lack_num = self.max_point_num - len(points_auto)
              
            # Random
            mask_indices = list(zip(*np.where(mask == 1)))
            if len(mask_indices) == 0:
                self._init_points_manual(k)
                continue
            elif len(mask_indices) <= lack_num:
                mask_indices_part = mask_indices
            else:
                mask_indices_part = random.sample(mask_indices, lack_num)
            for mask_index in mask_indices_part:
                point = np.array([mask_index[1], mask_index[0]]).astype(np.float32)
                if len(points) == 0:
                    points = point.reshape(-1, 2)
                else:
                    points = np.vstack([points, point])

            # Init points
            self.tracks[k].points = points

    def init_points_manual(self):
        for k in range(len(self.tracks)):
            self._init_points_manual(k)

    def _init_points_manual(self, k):
        points = np.array([])
        self.tracks[k].points = []
        bbox = self.tracks[k].bbox

        if self.focus_point_manual == "head":
            head = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] * 0.1]
        elif self.focus_point_manual == "center":
            head = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] * 0.5]

        point = np.array(head).astype(np.float32)
        points = point.reshape(-1, 2)

        # Around the focus point
        interval = int(bbox[2] / 2 * self.r_ratio / self.interval_num)
        if interval > 0:
            for m in range(1, self.interval_num + 1):
                r = m * interval
                for n in range(self.K):
                    theta = 2 * math.pi * n / self.K
                    point = np.array(head) + np.array([r * math.cos(theta), r * math.sin(theta)])
                    point = point.astype(np.float32)
                    points = np.vstack([points, point])

        # Init points
        self.tracks[k].points = points

    def update(self):
        # 1. Update by points
        if self.metric == "flow":
            self.update_by_points()

        # 2. Update by det
        if self.reinit == 1:
            self.update_by_det()
            if self.metric == "flow":
                self.init_points()

    def update_by_points(self):
        points, point_ids = self.get_all_points()

        self.already_updated = 0
        if len(points) == 0:
            print("Skip optical flow (no points)")
            return
        else:
            new_points, st, err = cv2.calcOpticalFlowPyrLK(self.frame_gray_old, self.frame_gray, points, None, **self.lk_params)
            self.already_updated = 1

        st = np.array([int(i) for i in st])
        points = points[st==1]
        new_points = new_points[st==1]
        point_ids = point_ids[st==1]

        # Update
        for i in range(len(self.tracks)):
            if i not in point_ids:
                self.tracks[i].mark_deleted()
                continue
            points_each_track = np.array(points)[point_ids==i]
            self.tracks[i].points = points_each_track
            new_points_each_track = np.array(new_points)[point_ids==i]
            self.tracks[i].update_by_points(new_points_each_track)

            # Mark missed
            if len(new_points_each_track) <= 3: #
                self.tracks[i].mark_missed()
            else:
                if self.point_termi == "variance": # Variance of points
                    var = np.var(points_each_track, axis=0)
                    new_var = np.var(new_points_each_track, axis=0)
                    var_ratio = new_var / var
                    if var_ratio[0] > self.thre_var_ratio or var_ratio[1] > self.thre_var_ratio:
                        self.tracks[i].mark_missed()
#                        print("track_id", self.tracks[i].track_id, "mark missed", var_ratio[0], var_ratio[1])

                elif self.point_termi == "homograpy": # Determinant of homograpy matrix
                    M = cv2.getPerspectiveTransform(points_each_track[0:4],new_points_each_track[0:4])
                    determ = np.linalg.det(M[0:2,0:2])
                    if abs(determ) < 1/self.thre_homo or self.thre_homo < abs(determ):
                        self.tracks[i].mark_missed()
#                        print("track_id", self.tracks[i].track_id, "mark deleted", determ)

        # Actually delete tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def update_by_det(self):
        feature_tracks = []
        feature_dets = []

        if self.detect_method == "fairmot":
            dets, feature_dets = self.detector(self.frame)
            feature_dets = np.array(feature_dets).reshape(-1, 1, 128)
        elif self.detect_method == "maskrcnn" or self.detect_method == "centermask":
            dets, masks = self.detector(self.frame)
        elif self.detect_method == "read_gt":
            dets = read_gt(self.gt_path, self.frame_ind)
        elif self.detect_method == "read_det":
            dets = read_det(self.det_path, self.frame_ind, self.x_max, self.y_max, self.max_size, self.thre_conf)

        if len(dets) == 0:
            print("No detection is found")
            return
        
        # Split tracks into confirmed and unconfirmed
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 1. Assign by feature
        if self.metric == "feature":
            # Tracks
            for track_id in confirmed_tracks:
                feature_tracks.append(self.tracks[track_id].feature)
            # Dets
            if self.detect_method != "fairmot":
                for det_id, det in enumerate(dets):
                    top, bottom, left, right = calc_tblr(det, self.x_max, self.y_max)
                    feature = self.extractor([self.frame[top:bottom,left:right]])
                    feature_dets.append(feature)

            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment(self.tracks, dets, feature_tracks, feature_dets, self.max_dist_feature, self.max_age, 1, confirmed_tracks)

        else:
            matches_a, unmatched_tracks_a, unmatched_detections = [], confirmed_tracks, np.arange(len(dets))

        # 2. Assign by IoU
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment(self.tracks, dets, [], [], self.max_dist_iou, self.max_age, 0, iou_track_candidates, unmatched_detections)

        # Concat
        matches = matches_a + matches_b
        unmatched_tracks = unmatched_tracks_b

        # Matches
        for track_idx, detection_idx in matches:
            if self.metric == "feature":
                feature = feature_dets[detection_idx]
            else:
                feature = []
            if (self.detect_method == "maskrcnn") or (self.detect_method == "centermask"):
                mask = masks[detection_idx]
            else:
                mask = []
            self.tracks[track_idx].update_by_det(dets[detection_idx], feature, mask, self.already_updated)

        # Unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Unmatched detections
        for detection_idx in unmatched_detections:
            if self.metric == "feature":
                feature = feature_dets[detection_idx]
            else:
                feature = []
            if (self.detect_method == "maskrcnn") or (self.detect_method == "centermask"):
                mask = masks[detection_idx]
            else:
                mask = []
            self.add_track(dets[detection_idx], feature, mask)
