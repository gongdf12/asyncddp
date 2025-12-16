#!/bin/bash
set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
export BLUEFOG_WIN_ON_GPU=1
export BLUEFOG_LOG_LEVEL=TRACE

# 可选：按 rank 分日志
LOGDIR=logs
mkdir -p "$LOGDIR"

bfrun -np 3 \
  --network-interface eno1 \
  bash -c '
    RANK=${OMPI_COMM_WORLD_RANK:-0}
    LOGFILE='"$LOGDIR"'/rank_${RANK}.log
    echo "Start rank $RANK, log=$LOGFILE" >"$LOGFILE"
    exec python test_win_async.py \
      --cuda \
      --op put \
      --mode single \
      --iters 4 \
      --compute-iters 0 \
      --topo full \
      --dim 102 >>"$LOGFILE" 2>&1
  '
