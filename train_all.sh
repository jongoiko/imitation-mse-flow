#!/bin/bash

declare -a tasks=(
    "pusht"
    "robomimic/lift"
    "robomimic/can"
    "robomimic/square"
    "robomimic/tool_hang"
    "robomimic/transport"
)

for task in "${tasks[@]}"
do
    uv run src/imitation/train.py \
        --task $task \
        --pred-chunk-size 16 --exec-chunk-size 8 --obs-horizon 2 \
        --rot-to-6d True --use-ema True \
        --batch-size 256 \
        --lr 0.0001 --seed $1 --lr-warmup-steps 500 \
        --weight-decay 1e-6 \
        --eval-interval 200000 \
        --use-ema True \
        --num-epochs 4500 --eval-episodes 100 \
        --num-video-episodes 10 \
        --hidden-dims "(256, 512, 1024)" \
        policy:flow-policy --policy.architecture unet \
        --policy.conv-kernel-size 5 --policy.time-embed-dim 256
done
