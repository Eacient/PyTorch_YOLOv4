python resnet/train.py \
    --cfg resnet/models/resnet18t-ds-verse.yaml \
    --data data/mhist.yaml \
    --epochs 500 \
    --batch-size 64 \
    --img-size 224 \
    --warmup \
    --ema \
    --device 0
    --hyp resnet/config/adam_base.yaml \