
# 当不支持nccl时可以使用CPU完成通信
CUDA_VISIBLE_DEVICES=6,7,5  bfrun -np 3 --extra-mpi-flags " -x OMPI_MCA_coll_hcoll_enable=0   -x OMPI_MCA_coll=^hcoll  " \
      python /home/gdf/bluefog/examples/test_win_async.py