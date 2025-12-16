export BLUEFOG_WIN_ON_GPU=1
export BLUEFOG_LOG_LEVEL=TRACE
CUDA_VISIBLE_DEVICES=6,7,5,4  bfrun -np 4  --network-interface eno1 \
     python test_data.py 