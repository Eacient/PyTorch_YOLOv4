python train.py \
    --cfg models/yolov4s-mish.yaml \
    --data data/neu.yaml \
    --epochs 300 \
    --batch-size 16 \
    --img-size 200 200 \
    --name yolov4s-no-hsv \
    --device 0

python train.py \
    --cfg models/yolov4m-mish.yaml \
    --data data/neu.yaml \
    --epochs 500 \
    --batch-size 16 \
    --img-size 200 200 \
    --name yolov4m \
    --device 0

python train.py \
    --cfg models/yolov4l-mish.yaml \
    --data data/neu.yaml \
    --epochs 500 \
    --batch-size 16 \
    --img-size 200 200 \
    --name yolov4l \
    --device 0

python train.py \
    --cfg models/yolov4x-mish.yaml \
    --data data/neu.yaml \
    --epochs 500 \
    --batch-size 16 \
    --img-size 200 200 \
    --name yolov4x \
    --device 0

python train.py \
    --cfg models/yolov4s-mish.yaml \
    --data data/neu-abnormal.yaml \
    --epochs 300 \
    --batch-size 16 \
    --img-size 200 200 \
    --name yolov4s-abnormal \
    --device 0 \
    --single-cls

python train.py \
    --cfg models/yolov4s-mish.yaml \
    --data data/neu-abnormal.yaml \
    --epochs 300 \
    --batch-size 16 \
    --img-size 200 200 \
    --name yolov4s-abnormal \
    --device 0 \
    --single-cls

# single-cls
# no-hsv
# generate-abnormal
# depth-wise-conv
# prune
# dw + 

python train.py \
    --cfg models/yolov4s-ae.yaml \
    --data data/neu-aug-train.yaml \
    --epochs 300 \
    --batch-size 16 \
    --img-size 200 200 \
    --name yolov4s-ae-aug-train \
    --device 0 \
    --single-cls \
    --ae