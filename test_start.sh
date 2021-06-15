#!/usr/bin/env bash
iter=43000
NETWORK="anchorFree_relation"
MODELDIR="./test_relation/test-$NETWORK-`date +%Y-%m-%d-%H-%M-%S`"
mkdir -p "$MODELDIR"
LOGFILE="$MODELDIR/log-iter-$iter-$NETWORK-`date +%Y-%m-%d-%H-%M-%S`.log"
CUDA_VISIBLE_DEVICES=1 python evaluate.py CornerNet_Squeeze --testiter $iter 2>&1 | tee $LOGFILE