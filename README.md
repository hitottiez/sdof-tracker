# SDOF-Tracker

## Requirements
- Ubuntu (18.04)
- OpenCV (>= 3.2)

## Setup
### Option1: Use the pre-computed detection result (Easy)
Set the detection result (![MOT format](https://motchallenge.net/)) to arbitary directory.

### Option2: Use detectron2 for detection
Refer ![Detectron2](https://github.com/facebookresearch/detectron2).

### Option3: Use FairMOT for detection
Refer ![FairMOT](https://github.com/ifzhang/FairMOT).

## Dataset
- ![MOT16](https://motchallenge.net/data/MOT16/)
- ![MOT20](https://motchallenge.net/data/MOT20/)

## Run
```
cd src
```
Change options in opts.pys.

Run tracking.
```
python tracking.py
```

## TODO

## Reference
- https://github.com/nwojke/deep_sort
- https://github.com/ZQPei/deep_sort_pytorch

