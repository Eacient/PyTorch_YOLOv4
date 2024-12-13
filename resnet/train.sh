python resnet/train.py \
    --cfg resnet/models/resnet18.yaml \
    --data data/mhist.yaml \
    --epochs 100 \
    --batch-size 64 \
    --img-size 224 \
    --device 0
    --warmup \
    --ema \