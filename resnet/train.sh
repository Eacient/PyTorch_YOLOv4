python swin/train.py \
    --cfg swin/models/swin-t-gp.yaml \
    --data data/mhist.yaml \
    --epochs 500 \
    --batch-size 64 \
    --img-size 224 \
    --warmup \
    --ema \
    --device 0
    --hyp resnet/config/adam_base.yaml \