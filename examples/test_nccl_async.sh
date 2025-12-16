export BLUEFOG_WIN_ON_GPU=1
# export OMPI_MCA_opal_cuda_support=1        # 关闭 MPI 的 CUDA 支持
# export OMPI_MCA_btl=self,tcp                   # 不用 vader/smcuda，只用 self+tcp
# CUDA_VISIBLE_DEVICES=6,7,5  bfrun -np 3 --extra-mpi-flags " -x OMPI_MCA_coll_hcoll_enable=0   -x OMPI_MCA_coll=^hcoll  " \
#       python /home/gdf/bluefog/examples/test_win_async.py --cuda --mode pingpong
CUDA_VISIBLE_DEVICES=6,7,5  bfrun -np 3  --network-interface eno1 \
     python test_async.py # > run1.log 2>&1