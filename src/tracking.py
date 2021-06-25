import numpy as np
import cv2
import os
import sys
import pickle
import datetime
import motmetrics as mm
from pytz import timezone
from tracker import Tracker
from opts import Opts

sys.path.append("../../FairMOT/src/lib")
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


# Read opts
opts = Opts()
opts.check_params()

# Save opts as pickle
#dtime = datetime.datetime.now(timezone("Asia/Tokyo"))
#output_opts_dir = "{}/opts/{}".format(opts.output_root_dir, opts.dataset)
#os.makedirs(output_opts_dir, exist_ok=True)
#output_opts_path = "{}/opts_{}.pickle".format(output_opts_dir, dtime)
#with open(output_opts_path, "wb") as fp:
#    pickle.dump(opts, fp)

# Prepare
color = np.random.randint(0, 255, (100000, 3)) # Create some random colors
accs = [] # Accuracies
total_frame_num = 0
total_time = 0

# Open file
os.makedirs(opts.output_speed_dir, exist_ok=True)
out_speed = open("{}/speed.csv".format(opts.output_speed_dir), "w")
out_speed.write("sequence,msec,fps,frame_num\n")
os.makedirs(opts.output_motmetrics_dir, exist_ok=True)

for seq_name in opts.seq_names:
    print(seq_name)

    # Input path
    det_path = "{}/{}/det/det.txt".format(opts.input_root_dir, seq_name)
    gt_path = "{}/{}/gt/gt.txt".format(opts.input_root_dir, seq_name)
    input_image_dir = "{}/{}/img1".format(opts.input_root_dir, seq_name)

    # Output path
    output_video_dir = "{}/videos".format(opts.output_root_dir)
    output_image_dir = "{}/images/{}/{}".format(opts.output_root_dir, opts.dataset, seq_name)
    output_track_dir = "{}/tracks".format(opts.output_root_dir)
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_track_dir, exist_ok=True)

    # Open file
    output_track_path = "{}/{}.txt".format(output_track_dir, seq_name)
    out_track = open(output_track_path, "w")

    # Init tracker
    tracker = Tracker(det_path, gt_path, opts.max_age, opts.max_dist_iou, opts.max_dist_feature, opts.n_init, opts.reinit_interval, opts.metric, opts.detect_method, opts.point_termi, opts.start_ind, opts.max_size, opts.thre_conf, opts.thre_var_ratio, opts.thre_homo, opts.point_detect, opts.focus_point_manual, opts.focus_point_auto, opts.use_mask, opts.head_detect, opts.r_ratio, opts.interval_num, opts.K, opts.max_point_num, opts.shi_tomasi, opts.feature_params, opts.lk_params)

    frame_ind = opts.start_ind
    timer = Timer()

    while(1):
        if frame_ind == 1 or frame_ind % opts.show_interval == 0:
            print("Frame: {:06d}".format(frame_ind))

        # Read frame
        frame = cv2.imread("{}/{:06d}.jpg".format(input_image_dir, frame_ind))

        # Update frame information
        timer.tic()
        if frame is None or frame_ind > opts.end_ind:
            print("Finish")
            break
        else:
            tracker.update_frame(frame_ind, frame)
            tracker.update_status_every()

        # Update tracker
        tracker.update()
        timer.toc()

        # Draw and write
        for track in tracker.tracks:
#            if track.state == TrackState.Confirmed:
            if 1:
                track_id = track.track_id
                bbox = [int(i) for i in track.bbox]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color[track_id].tolist(), thickness=2)
                cv2.putText(frame, str(track_id), (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, color[track_id].tolist(), thickness=2)
                out_track.write("{:d},{:d},{:0.2f},{:0.2f},{:0.2f},{:0.2f},-1,-1,-1,-1\n".format(frame_ind, track_id, bbox[0], bbox[1], bbox[2], bbox[3]))
                for point in track.points:
                    cv2.circle(frame, (point[0], point[1]), 3, color[track_id].tolist(), -1)

        # Save
        if opts.save_output_images:
            cv2.imwrite("{}/{:06d}.jpg".format(output_image_dir, frame_ind), frame)

        # Next frame
        frame_ind += 1

    # Release
    cv2.destroyAllWindows()
    out_track.close()

    # Convert output images to one video
    #cmd_str = 'ffmpeg -f image2 -i {}/%06d.jpg -b 5000k -c:v mpeg4 {}.{}.mp4'.format(opts.output_image_dir, opts.output_video_dir, seq_name)
    #os.system(cmd_str)

    # Speed
    out_speed.write('{},{:.2f},{:.2f},{:d}\n'.format(seq_name, max(1e-5, timer.average_time * 1000), 1. / max(1e-5, timer.average_time), frame_ind))
    total_time += timer.total_time

    # Evaluate
    if opts.eval_on:
        evaluator = Evaluator(opts.input_root_dir, seq_name, "mot")
        accs.append(evaluator.eval_file(output_track_path))

    # Total frame number
    total_frame_num += frame_ind

# Summarize speed
out_speed.write('total,{:.2f} (sec),{:.2f},{:d}\n'.format(total_time, 1. / (total_time / total_frame_num), total_frame_num))
out_speed.close()

# Summarize evaluation
if opts.eval_on:
    #mm.lap.default_solver = 'scipy'
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, opts.seq_names, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(opts.output_motmetrics_dir, 'summary.xlsx'))
