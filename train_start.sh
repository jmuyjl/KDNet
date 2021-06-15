#!/usr/bin/env bash
NETWORK="anchorFree_relation"
MODELDIR="./train_relation/log-$NETWORK-`date +%Y-%m-%d-%H`"
mkdir -p "$MODELDIR"
LOGFILE="$MODELDIR/log-$NETWORK-`date +%Y-%m-%d-%H-%M-%S`.log"
iter=5
CUDA_VISIBLE_DEVICES=2,3 python train.py --iter $iter CornerNet_Squeeze 2>&1 | tee $LOGFILE