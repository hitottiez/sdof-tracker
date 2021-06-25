import cv2

class Opts:
    # Dataset
    data = "train" # "train" or "test"
    dataset = "MOT20" # "MOT16" or "MOT20"

    if dataset == "MOT16":
        if data == "train":
            seq_names = ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]
            eval_on = 1
        elif data == "test":
            seq_names = ["MOT16-01", "MOT16-03", "MOT16-06", "MOT16-07", "MOT16-08", "MOT16-12", "MOT16-14"]
            eval_on = 0
    elif dataset == "MOT20":
        if data == "train":
            seq_names = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
#            seq_names = ["MOT20-05"]
            eval_on = 1
        elif data == "test":
            seq_names = ["MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"]
            eval_on = 0

#    dataset = "20201230_KEIRIN"
#    seq_names = ["00000"]

    # Path
    input_root_dir = "/mnt/disk/Dataset/tracking/{}/{}".format(dataset, data)
    output_root_dir = "/mnt/disk/Dataset/tracking/{}/results_{}".format(dataset, data)
    output_speed_dir = "{}/speed".format(output_root_dir)
    output_motmetrics_dir = "{}/motmetrics".format(output_root_dir)

    # Tracker
    max_dist_iou = 0.7 # 0.7
    max_dist_feature = 0.3 # 0.3
    max_age = 5 # 5, 10 ##### C
    n_init = 1
    reinit_interval = 25 # 1, 5, 10, 15, 20, 25
    metric = "flow" # "feature" or "flow"
    detect_method = "maskrcnn" # "read_det" or "read_gt" or "maskrcnn" or "centermask" or "fairmot"
    point_termi = "variance" # "variance" or "homography"
    show_interval = 100
    start_ind = 1 # start from 1
    end_ind = 100000000
    save_output_images = 1
    max_size = 1000
    thre_conf = 0.2 ## FairMOT have to set in command line!! 0.2 ## read_det have to set to -2!!
    thre_var_ratio = 10000000 # 10, 10000000 ##### T
    thre_homo = 100000

    # Point detection
    point_detect = "auto" # "auto" or "manual"
    head_detect = 0 # 1 or 0

    ## Manual
    r_ratio = 0.3
    interval_num = 2
    K = 8
    focus_point_manual = "head" # "head" or "center"

    ## Auto
    ### Mask
    focus_point_auto = "head" # "head" or "center" or "none"
    use_mask = 0 # 1 or 0 ##### S
    max_point_num = 10

    ### Shi-Tomasi
    shi_tomasi = 0 # 1: shi_tomasi (slow) or 0: random
    feature_params = dict(maxCorners = 10,
                       qualityLevel = 0.001,
                       minDistance = 1,
                       blockSize = 7)

    # Lucas kanade optical flow
    lk_params = dict(winSize = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def check_params(self):
        print("eval_on: {:d}".format(self.eval_on))
