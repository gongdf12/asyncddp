import os
import copy
import math
from loguru import logger
from functools import partial
from typing import Callable, Iterator, List, Optional, Tuple, cast,Dict
import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from torch.optim import Optimizer
from torch.distributed import Work
from torch import GradScaler
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle # 动态控制钩子的生命周期
from torch.optim.lr_scheduler import LRScheduler
import bluefog.torch as bf
from bluefog.common import topology_util  # 应用bluefog的拓扑写法，暂时不写
import networkx as nx



# import math
# from typing import Iterator, List
# import torch
from torch.utils.data import BatchSampler, Dataset


class LocalStepRandomBatchSampler(BatchSampler):
    """每个 rank 本地随机批采样，local_step 决定当前用哪一轮/哪一批。

    - 每一轮用一个新的全局乱序 perm。
    - 一轮内 num_batches = ceil(N / batch_size)，逐批顺序取。
    - local_step 跨轮累加，随机性只依赖本 rank 的 seed + local_step。
    - 这个是异步的数据分配类
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        base_seed: int,
        rank: int = 0,
        drop_last: bool = False,
    ) -> None:
        assert batch_size > 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = base_seed
        self.rank = rank

        self.N = len(dataset)
        self.num_batches_per_round = self._compute_num_batches()
        self.local_step = 0  # 持久化的本地步数
        self._perm: List[int] = []

    def _compute_num_batches(self) -> int:
        if self.drop_last:
            return self.N // self.batch_size
        return math.ceil(self.N / self.batch_size)

    def _ensure_perm_for_round(self, round_id: int) -> None:
        # 每一轮生成一个新的乱序，seed 只依赖 base_seed/rank/round_id
        g = torch.Generator()
        seed = self.base_seed + self.rank * 10000 + round_id
        g.manual_seed(seed)
        self._perm = torch.randperm(self.N, generator=g).tolist()

    def __iter__(self) -> Iterator[List[int]]:
        # DataLoader 每次 __iter__ 调用时，我们按“当前 round”生成一轮 batch
        round_id = self.local_step // self.num_batches_per_round
        idx_in_round_start = self.local_step % self.num_batches_per_round

        self._ensure_perm_for_round(round_id)

        for k in range(idx_in_round_start, self.num_batches_per_round):
            start = k * self.batch_size
            end = min(start + self.batch_size, self.N)

            if self.drop_last and end - start < self.batch_size:
                continue

            batch_indices = self._perm[start:end]
            self.local_step += 1
            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches_per_round








# from .topo import TopologyReg, Topology

"""Data type for the optimizer function"""
OPTIM_FN_TYPE = Callable[[List[Tuple[str, Tensor]]], Optimizer]


"""Data type for the learning rate scheduler function"""
LR_SCHEDULER_FN_TYPE = Callable[[Optimizer], LRScheduler] #提示说明函数的参数类型与返回类型callable【【参数】，返回】

"""topology"""
# TOPOLOGY = Callable[[nx.DiGraph],]




class DecentralizedDataParallel(Module):
    """Decentralized data parallel wrapper for PyTorch module

    1. The wrapper places hooks during the backward pass to trace the order of used parameters in the first iteration, and \
    2. Split the parameters into buckets and create optimizers and LR schedulers for each bucket, \
        Add hooks on the last parameter of each bucket to perform the bucket-wise update and communication, \
    3. During the backward passes in the training loop, the hooks are triggered to perform the bucket-wise update and communication

    :Warning: The wrapper currently does not support "channels_last" memory format

    :Warning: The wrapper assumes that the parameter will only be used once in the backward pass

    Args:
        model (Module): PyTorch module to be wrapped
        optim_fn (OPTIM_FN_TYPE): Function to create the optimizer, which takes a list of tuples of parameters and their names
        lr_scheduler_fn (Optional[LR_SCHEDULER_FN_TYPE], optional): Function to create the learning rate scheduler, \
            which takes the optimizer as input. Defaults to None.
        topology (str, optional): Topology of the decentralized communication graph. Defaults to 'complete'.
        scaler (Optional[GradScaler], optional): Gradient scaler for mixed precision training. Defaults to None.
        grad_clip_norm (float, optional): Gradient clipping norm, set to 0.0 if no gradient clipping is applied. Defaults to 0.0.
        param_as_bucket_view (bool, optional): Whether to use the parameter as a view of part of the contiguous buffer. Defaults to True.
        sync_buffer_in_global_avg (bool, optional): Whether to synchronize the float buffers in the global average. Defaults to False.
        bucket_size_in_mb (int, optional): Size of the bucket in MB. Defaults to 25 MB.
        local_world_size (Optional[int], optional): Provide the local world size if not using the environment variable. Defaults to None.
    """

    """Buffer data types that need to be synchronized in global average"""
    FLOAT_DTYPES = [torch.float16, torch.float32, torch.float64]
    # 模型，优化器，学习率调度器，拓扑结构     #Gradscaler是混合精度训练的梯度缩放器类型
    def __init__(self,
                 model: Module,
                 optim_fn: OPTIM_FN_TYPE,
                 lr_scheduler_fn: Optional[LR_SCHEDULER_FN_TYPE] = None,#可选类型 可以是None或LR_SCHEDULER_FN_TYPE
                 topology = None,
                 sync = True,
                 scaler: Optional[GradScaler] = None,
                 grad_clip_norm: float = 0.0, #梯度裁剪范数，会缩放梯度
                 param_as_bucket_view: bool = True,
                 sync_buffer_in_global_avg: bool = False,#同步全局缓冲区
                 bucket_size_in_mb: int = 25, #桶大小，25mb
                 _local_world_size: Optional[int] = None):
        
        super(DecentralizedDataParallel, self).__init__()
#         assert bf.is_available() and bf.is_initialized(), 'Distributed environment is not initialized'

        self._model = model.cuda() if torch.cuda.is_available() else model
        self._optim_fn = optim_fn
        self._lr_schd_fn = lr_scheduler_fn
        self._scaler = scaler
        self._grad_clip_norm = grad_clip_norm
        self._param_as_bucket_view = param_as_bucket_view
        self._sync_buffer_in_global_avg = sync_buffer_in_global_avg
        self._bucket_size = bucket_size_in_mb * 1024 * 1024
        self._local_world_size = _local_world_size if _local_world_size is not None else int(os.environ.get('LOCAL_WORLD_SIZE', 1))

        # get the rank and world size
        self._rank = bf.rank()
        self._world_size = bf.size()
        """当前机器内本地 rank → bf.local_rank(),
           当前机器内本地进程数 → bf.local_size()
           机器 id（按机器编号）→ bf.machine_rank()
           机器总数 → bf.machine_size()  
        """
        # check if the model is with "channels_last" memory format
        if self._check_channels_last():
            if self._rank == 0:
                logger.debug(f'The model is with "channels_last" memory format')

        if self._rank == 0:
            logger.debug(f'Initializing Decentralized Data Parallel')
            logger.debug(f'Rank: {self._rank}, Local World Size: {self._local_world_size}, World Size: {self._world_size}, Topology: {topology}')

        # model parameters
        self._params: List[Tensor] = list([x for _, x in self._model.named_parameters() if x.requires_grad])
        self._param_names: List[str] = list([n for n, x in self._model.named_parameters() if x.requires_grad])

        # trace hooks and traced parameter ids
        self._trace_hooks: List[RemovableHandle] = []
        self._traced_param_ids: List[int] = []

        self._step: int = 0
        self._comm_ops: List[Optional[Work]] = []  #Work 对象表示 PyTorch 分布式包中待处理的异步操作的句柄。它由非阻塞集体通信操作返回，例如 dist.all_reduce(tensor, async_op=True)
        
        self._ddp_hooks: List[RemovableHandle] = []
        self._param_buckets: List[List[Tensor]] = []
        self._param_blocks: List[Tensor] = []
        self._comm_buffers: List[List[Tensor]] = []
        self._comm_blocks: List[Tensor] = []

        # per-bucket, per-neighbor window cache for async win ops
        self._neigh_cache: List[Dict[int, Tensor]] = []
        self._neigh_versions: List[Dict[int, int]] = []

        self.sync = sync
        
        self._pp_state = [] # 這個列表用來存放窗口通信信息 每個bucket：{name，buf，prev_handle,iter}
        self._win_names = []
        # Optimizer and LR scheduler
        self._optims: List[Optimizer] = []
        self._lr_schedulers: List[Optional[LRScheduler]] = []

        # initialize the topology
#         self._topo: Topology = TopologyReg.registry[topology](self._local_world_size)
        self._topo = topology
        self.weights = {'self_weight':1.0}
        # create hooks to trace the used parameters in backward
        self._create_trace_hooks()

        # sync the parameters at the start
        self._sync_at_start()

        # flag for gradient accumulation
        self._is_grad_accum_enable: bool = False

        # flag for initializing the parameters
        self._initialized: bool = False


    def _check_channels_last(self) -> bool:
        """Check if the model is with "channels_last" memory format

        Returns:
            bool: True if the model is with "channels_last" memory format
        """
        if any([x.is_contiguous(memory_format=torch.channels_last) and (not x.is_contiguous()) for x in self._model.parameters() if len(x.shape) == 4]):
            return True
            # 检查是否为channel-last 存储，结合amp，NHWC (channels_last)	数据按 [批, 高, 宽, 通道] 存储	AMP训练、TensorCore加速
        return False


    def _create_trace_hooks(self):
        """Create hooks to trace the order of used parameters in backward pass
        #跟踪反向传播过程
        """
        for pid, param in enumerate(self._params):     # 遍历self._params的同时获取元素的索引值
            self._trace_hooks.append(
                #register_post_accumulate_grad_hook返回对象为removeablehook
                param.register_post_accumulate_grad_hook( #操作梯度，创建钩子，对梯度积累完成后调用，但是在参数更新之前，它的参数是函数，函数是对梯度处理。它的返回可用于移除钩子
                    partial(     #固定函数参数从而生成新的函数
                        lambda data, pid: self._trace_fn(data, pid), #判断pid  这个梯度是否被使用，使用报错，否则加入列表
                        pid=pid # lambda data, pid: self._trace_fn(data, pid)中的pid被固定
                    )
                )
            )
    
    @torch.no_grad()
    def _sync_at_start(self):
        """Broadcast the parameters of worker 0 to all other workers at the start
        """
        for param in self._params: # 开始的时候广播梯度
            h=bf.broadcast_(param, 0)
    

    def set_accumulate_grad(self, enable: bool = True):
        """Set the gradient accumulation mode

        Args:
            enable (bool, optional): Whether to accumulate the gradients. Defaults to True.
        """
        self._is_grad_accum_enable = enable #设置梯度累计模式

    
    """Hook functions"""

    @torch.no_grad()
    def _trace_fn(self, _: Tensor, pid: int): #判断梯度是否使用
        """Hook function to trace the order of used parameters in backward pass

        Args:
            _ (Tensor): corresponding tensor (not used)
            pid (int): parameter id
        
        Raises:
            AssertionError: The parameter is used more than once in the backward pass
        """
        if self._is_grad_accum_enable:
            return
        assert not (pid in self._traced_param_ids), 'The parameter is used more than once in the backward pass'
        self._traced_param_ids.append(pid) #使用列表存储可以用的id
    def _over_communication(self): #重点检查这个
        bf.barrier()
        if self._pp_state ==[]:
            return
        if not self.sync:
            # 先处理之前没有处理完的通信
            print('处理之前的通信')
            for i in range(len(self._pp_state)):
                handle= self._pp_state[i]
                if handle is not None:
                    print(bf.win_poll(handle))
                    if not bf.win_poll(handle): #返回true，表示通信完成
                        bf.win_wait(handle)
                        self._pp_state[i]=None
                    else:
                        # 这个handle已经不存在，
                        self._pp_state[i]=None
                else:
                    pass
        else:
            print('处理之前的handle')
            for i in range(len(self._comm_ops)):
                    if self._comm_ops[i] is not None:        
                        self._comm_blocks[i]=bf.synchronize(self._comm_ops[i])
                        self._comm_ops[i] = None
            self._comm_ops = [None for _ in range(len(self._param_buckets))]
            
    def _create_win(self,):
        bf.barrier()
        for b in range(len(self._param_blocks)):
            name = f'back_win_bucket_{b}'
            self._win_names.append(name)
            # 用通信块创建窗口（所有 rank 上形状一致）
            base = self._comm_blocks[b]
            logger.debug(f'窗口的大小为：{base.shape}，类型{base.dtype}')
            ok = bf.win_create(base, name=name, zero_init=True)
            if not ok and bf.rank() == 0:
                raise RuntimeError(f'win_create failed for {name}')
        logger.debug(f'win_create done')
        bf.barrier()

    """topology """
    def set_topology(self , topology):
            
        self._topo = topology['topology']
        bf.set_topology(topology['topology']) # 只有调用了这个函数后节点才知道邻居，才知道和谁通信

        self_weight = self._topo[self._rank].get(self._rank, {}).get('weight', 1.0)

        # 出/入邻居
        out_ranks = bf.out_neighbor_ranks()
        in_ranks  = bf.in_neighbor_ranks()
        # print(f'set topo out_ranks:{out_ranks}, in_ranks:{in_ranks}')
        # 出/入邻居权值（字典形式，key 为邻居 rank）
        out_weights = {v: self._topo[self._rank][v].get('weight', 1.0) for v in out_ranks}
        in_weights  = {u: self._topo[u][self._rank].get('weight', 1.0) for u in in_ranks}
        # logger.debug(f'out_weights:{out_weights}')
        # 若你需要与 out_ranks/in_ranks 对齐的列表，可这样取：
        out_weights_list = [out_weights[v] for v in out_ranks]
        in_weights_list  = [in_weights[u]  for u in in_ranks]

        self.weights.update({
        'self_weight': self_weight,
        'out_neighbor_ranks': out_ranks,
        'in_neighbor_ranks': in_ranks,
        'out_neighbor_weights': out_weights,   # 或 out_weights_list
        'in_neighbor_weights': in_weights,     # 或 in_weights_list
        })
        logger.debug(f'set topo done')    
    @torch.no_grad()
    def _ddp_fn(self, _: Tensor, bucket_id: int): # 利用钩子实现通信和信息传递
        """Hook function to perform the bucket-wise update and communication

        Args:
            _ (Tensor): corresponding tensor (not used)
            bucket_id (int): bucket id
        """

        # skip the update and communication if the model is accumulating gradients
        if self._is_grad_accum_enable:
            return
       
        # perform the bucket-wise update and communication when all gradients in the bucket are accumulated
        comm_op = self._comm_ops[bucket_id] #通信部分 ，是bucket_id 的Work类
        if comm_op is not None:
            # wait for the communication from the last iteration
           
            out = bf.synchronize(comm_op)
            self._comm_blocks[bucket_id].copy_(out)
            self._comm_ops[bucket_id] = None #注销该桶的通信类对象

            # topo ，使用nx 构造的topo图，此处钩子先考虑无向图的情况
            G = self._topo
            self_weight = G[self._rank][self._rank].get('weight', 1.0)
            # 此处更新
            if hasattr(self._optims[bucket_id], 'pre_average_hook'): #判断优化器中有没有‘pre—average-hook’
                self._optims[bucket_id].pre_average_hook(edge, weight) # type: ignore

            # replace the local model with the mixed model
            if self._param_as_bucket_view:#梯度是否是以桶的形式显示  
                
                # self._param_blocks[bucket_id].mul_(self_weight - (1 - self_weight) / (len(self.weights['in_neighbor_weights']) - 1))
                # # 张量乘法，更新桶梯度
                # self._param_blocks[bucket_id].add_(self._comm_blocks[bucket_id])
                # # 张量加法
                self._param_blocks[bucket_id].data.copy_(out.data)
            else:
                 torch._foreach_copy_(self._param_buckets[bucket_id], out)
        # perform local update
        if self._scaler: #Gradscaler 混合梯度训练
            if self._grad_clip_norm > 0:
                self._scaler.unscale_(self._optims[bucket_id])#将之前由GradScaler放大的梯度还原为原始值
               
                # 4. 在反缩放后的梯度上裁剪
                torch.nn.utils.clip_grad_norm_(self._param_buckets[bucket_id], self._grad_clip_norm)
            
            # 更新参数
            self._scaler.step(self._optims[bucket_id])
            if bucket_id == len(self._param_buckets) - 1:
                self._scaler.update()  #调整缩放因子
        else:
            if self._grad_clip_norm > 0:
                #裁剪
                torch.nn.utils.clip_grad_norm_(self._param_buckets[bucket_id], self._grad_clip_norm)
           
            # 更新
            self._optims[bucket_id].step()
        self._optims[bucket_id].zero_grad()#梯度清零操作

        if self._lr_schedulers[bucket_id] is not None:
            scheduler = cast(LRScheduler, self._lr_schedulers[bucket_id])
            scheduler.step()

        # launch the next communication after updating the weights
        #为下一轮通信准备，在更新weight之后
        if self._param_as_bucket_view:
            self._comm_blocks[bucket_id].copy_(self._param_blocks[bucket_id])
        else:
            torch._foreach_copy_(self._comm_buffers[bucket_id], self._param_buckets[bucket_id])

 

        self._comm_ops[bucket_id] = bf.neighbor_allreduce_nonblocking(
            self._comm_blocks[bucket_id],
            self_weight = self.weights['self_weight'],
            src_weights=self.weights['in_neighbor_weights'],
            dst_weights=self.weights['out_neighbor_weights'],
            name =f'back_comm_bucket_{bucket_id}'
            
        )#返回一个handle
    @torch.no_grad()
    def _ays_ddp_fn(self, _: Tensor, bucket_id: int):
            # 这个钩子进行异步通信
          
        if self._is_grad_accum_enable:
            return
        
        '''有写(put)与更新的底层互斥后(保证写的时候内容不改变),不需要考虑pyhton层面的等待与互斥机制,
        底层的互斥是基于内存管理逻辑，即同一时间只有一个修改操作在某一个内存上，
        这个互斥的细度为每个窗口的某个rank的内存空间。python层面只做三件事-接收-更新-发送,
        保留通信桶与参数桶，分离通信更新与梯度更新'''
        y=bf.win_update(self._win_names[bucket_id], 
                      self_weight=self.weights['self_weight'],
                      neighbor_weights=self.weights['in_neighbor_weights'],
                      require_mutex=True,
                      clone=True) #这个只进行通信的更新，没有原地修改了comm_block的值。
        
        # 通信完成，利用通信数据更新参数
        # replace the local model with the mixed model
        if self._param_as_bucket_view:#梯度是否是以桶的形式显示  
            
            
            self._param_blocks[bucket_id].data.copy_(y.data)
            # 张量加法
        else:
            # torch._foreach_mul_(self._param_buckets[bucket_id], self_weight - (1 - self_weight) / (len(self.weights['in_neighbor_weights']) - 1))
            torch._foreach_copy_(self._param_buckets[bucket_id], y)
    
        if self._scaler: #Gradscaler 混合梯度训练
            if self._grad_clip_norm > 0:
                self._scaler.unscale_(self._optims[bucket_id])#将之前由GradScaler放大的梯度还原为原始值
               
                # 4. 在反缩放后的梯度上裁剪
                torch.nn.utils.clip_grad_norm_(self._param_buckets[bucket_id], self._grad_clip_norm)
            
            # 更新参数
            self._scaler.step(self._optims[bucket_id])
            if bucket_id == len(self._param_buckets) - 1:
                self._scaler.update()  #调整缩放因子
        else:
            if self._grad_clip_norm > 0:
                #裁剪
                torch.nn.utils.clip_grad_norm_(self._param_buckets[bucket_id], self._grad_clip_norm)
           
            # 更新
            self._optims[bucket_id].step()
        self._optims[bucket_id].zero_grad()#梯度清零操作

        # 3) 准备下一轮：根据当前参数更新通信块，然后通过窗口异步发送
        self_w = self.weights['self_weight']
        if self._param_as_bucket_view:
            self._comm_blocks[bucket_id].copy_(self._param_blocks[bucket_id])
        else:
            torch._foreach_copy_(self._comm_buffers[bucket_id], self._param_buckets[bucket_id])
        # logger.debug(f'准备发送的张量大小：{self._comm_blocks[bucket_id].shape}，类型{self._comm_blocks[bucket_id].dtype}')
        handle=bf.win_put_nonblocking(self._comm_blocks[bucket_id],
                                name=self._win_names[bucket_id],
                               
                                require_mutex=True) # self_weight 进行的是通信完成后的加权缩放
        
        self._pp_state[bucket_id]=handle
       

        
    @torch.no_grad()
    def _initialize_params(self):
        """Initialize the parameter buckets and communication buffers

        Raises:
            RuntimeError: Number/Order of elements in used parameters is different on different nodes
        """

        # verify the number of elements and the order of the parameters on different nodes are the same
        verify = [[(i, self._params[i].numel()) for i in self._traced_param_ids]]
        result = [[(0, 0)]] if self._rank != 0 else verify
        bf.broadcast_object_list(result, src=0,name='verify_meta')
        if not all([x == y for x, y in zip(verify[0], result[0])]):
            raise RuntimeError('Number/Order of elements in used parameters is different on different nodes')
    
        # remove the trace hooks
        for hook in self._trace_hooks:
            hook.remove()
        del self._trace_hooks

        # split the parameters into roughly equal-size buckets, and register hooks on the last parameter of each bucket
        start = 0
        size = 0
        for i in range(len(self._traced_param_ids)):
            size += self._align(self._params[self._traced_param_ids[i]].numel()) * self._params[self._traced_param_ids[i]].element_size()
            if (size >= self._bucket_size) or (i == len(self._traced_param_ids) - 1):
                # register hooks on the last parameter of each bucket, passing the bucket id
                self._ddp_hooks.append(
                    self._params[self._traced_param_ids[i]].register_post_accumulate_grad_hook(
                        partial(
                            lambda data, bucket_id: self._ddp_fn(data, bucket_id) if self.sync else self._ays_ddp_fn(data, bucket_id) ,
                            bucket_id=len(self._ddp_hooks)
                        )
                    )
                )
                self._param_buckets.append([self._params[j] for j in self._traced_param_ids[start:i+1]])
                param_names = [self._param_names[j] for j in self._traced_param_ids[start:i+1]]
               
                # create optimizer and learning rate scheduler for parameters in each bucket
                self._optims.append(self._optim_fn(list(zip(param_names, self._param_buckets[-1]))))
                self._lr_schedulers.append(self._lr_schd_fn(self._optims[-1]) if self._lr_schd_fn is not None else None)
                size = 0
                start = i + 1

        size_dict = {}

        for i in range(len(self._param_buckets)):
            total_size = sum([self._align(p.numel()) for p in self._param_buckets[i]])

            # make sure the total size is unique for each bucket \
            # (not necessary, but make sure the communication operations are unique for each bucket with negligible overhead)
            while total_size in size_dict:
                total_size += 32
            size_dict[total_size] = True

            # create the communication buffer for each bucket
            comm_block = torch.zeros(total_size,
                                     device=self._param_buckets[i][0].device,
                                     requires_grad=False,
                                     dtype=self._param_buckets[i][0].dtype)

            if self._param_as_bucket_view:
                # create contiguous blocks for each bucket, and let the parameters be views of the fragments of the block
                self._param_blocks.append(torch.zeros(total_size,
                                                      device=self._param_buckets[i][0].device,
                                                      requires_grad=True,
                                                      dtype=self._param_buckets[i][0].dtype))
                start = 0
                for j in range(len(self._param_buckets[i])):
                    size = self._param_buckets[i][j].numel()
                    if (len(self._param_buckets[i][j].shape) == 4) and self._param_buckets[i][j].is_contiguous(memory_format=torch.channels_last) \
                        and (not self._param_buckets[i][j].is_contiguous()):
                        # permute the tensor to the channels_last format
                        self._param_blocks[-1].narrow(0, start, size).copy_(self._param_buckets[i][j].permute(0, 2, 3, 1).view(-1))
                        self._param_buckets[i][j].data = self._param_blocks[-1].narrow(0, start, size).view(
                            (self._param_buckets[i][j].shape[0],
                             self._param_buckets[i][j].shape[2],
                             self._param_buckets[i][j].shape[3],
                             self._param_buckets[i][j].shape[1])
                        ).permute(0, 3, 1, 2)
                        assert self._param_buckets[i][j].is_contiguous(memory_format=torch.channels_last)
                        assert not self._param_buckets[i][j].is_contiguous()
                    else:
                        # otherwise, copy the tensor directly
                        assert self._param_buckets[i][j].is_contiguous()
                        self._param_blocks[-1].narrow(0, start, size).copy_(self._param_buckets[i][j].view(-1))
                        self._param_buckets[i][j].data = self._param_blocks[-1].narrow(0, start, size).view_as(self._param_buckets[i][j])
                    start += self._align(size)

            self._comm_blocks.append(comm_block)
            start = 0
            self._comm_buffers.append([])
            for j in range(len(self._param_buckets[i])):
                size = self._param_buckets[i][j].numel()
                if (len(self._param_buckets[i][j].shape) == 4) and self._param_buckets[i][j].is_contiguous(memory_format=torch.channels_last) \
                    and (not self._param_buckets[i][j].is_contiguous()):
                    # permute the tensor to the channels_last format
                    self._comm_buffers[-1].append(comm_block.narrow(0, start, size).view(
                        (self._param_buckets[i][j].shape[0],
                         self._param_buckets[i][j].shape[2],
                         self._param_buckets[i][j].shape[3],
                         self._param_buckets[i][j].shape[1])
                    ).permute(0, 3, 1, 2))
                else:
                    self._comm_buffers[-1].append(comm_block.narrow(0, start, size).view_as(self._param_buckets[i][j]))
                start += self._align(size)

                # attach the communication buffer to the parameter for "pre_average_hook" in the optimizer
                if hasattr(self._optims[i], 'pre_average_hook'):
                    setattr(self._param_buckets[i][j], 'comm_buffer', self._comm_buffers[-1][-1])
            
            # initialize the communication buffer with the initial parameters
            torch._foreach_copy_(self._comm_buffers[-1], self._param_buckets[i])

        self._comm_ops = [None] * len(self._param_buckets)
        self._pp_state = [None] * len(self._param_buckets)
        self._create_win()
        if not self.sync:
            bf.set_skip_negotiate_stage(True) #异步设置这个防止卡死
    def _align(self, size: int):
        """Align the size to 128-byte boundary
        """
        return math.ceil(size / 32) * 32


    """Delegation functions"""

    def train(self, mode: bool = True):
        """Set the module in training mode

        Args:
            mode (bool, optional): Whether to set the module in training mode. Defaults to True.
        """
        self._model.train(mode)
        return self
    
    def eval(self):
        """Set the module in evaluation mode"""
        self._model.eval()
        return self

    def forward(self, *args, **kwargs):
        """Forward pass of the model
        """
        if (self._step == 1) and (not self._initialized):
            self._initialized = True
            # initialize the parameters and communication buffers
            self._initialize_params()

            # manually trigger the communications for the first iteration only
            with torch.no_grad():

                for i in range(len(self._param_buckets)):
                    # optionally call the pre_average_hook for optimizers using the communication information
                    if hasattr(self._optims[i], 'pre_average_hook'):
                        self._optims[i].pre_average_hook(edge, weight) # type: ignore

                    # update parameters and launch the first communication
                    if self._scaler:
                        if self._grad_clip_norm > 0:
                            self._scaler.unscale_(self._optims[i])
                            torch.nn.utils.clip_grad_norm_(self._param_buckets[i], self._grad_clip_norm)
                        self._scaler.step(self._optims[i])
                        if i == len(self._param_buckets) - 1:
                            self._scaler.update()
                            # TODO: synchronize the scaler state across all workers?
                    else:
                        if self._grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self._param_buckets[i], self._grad_clip_norm)
                        self._optims[i].step()
                    self._optims[i].zero_grad()
                    if self._lr_schedulers[i] is not None:
                        scheduler = cast(LRScheduler, self._lr_schedulers[i])
                        scheduler.step()
                    
                    # launch the first communication
                    if self._param_as_bucket_view:
                        self._comm_blocks[i].copy_(self._param_blocks[i])
                    else:
                        torch._foreach_copy_(self._comm_buffers[i], self._param_buckets[i])
                    
                    self._comm_blocks[i].mul_((1 - self.weights['self_weight']) / (len(self.weights['in_neighbor_weights']) - 1))
                    comm_op = bf.neighbor_allreduce_nonblocking(
                        self._comm_blocks[i],
                        self_weight = self.weights['self_weight'],
                        src_weights=self.weights['in_neighbor_weights'],
                        dst_weights=self.weights['out_neighbor_weights'],
                        name =f'fore_comm_bucket_{i}'
                    )
                    self._comm_ops[i] = comm_op

                    
                    # wait for the communication to finish to fully synchronize the workers
                  #   assert comm_op is not None
                    self._comm_blocks[i]=bf.synchronize(comm_op)
                    self._comm_ops[i] = None
                   #   comm_op.wait()

        if self._model.training and (not self._is_grad_accum_enable):
             
            self._step += 1

        with torch.autograd.profiler.record_function("DecentralizedDataParallel.forward"):
            # logger.debug('前向传播中')
            output = self._model(*args, **kwargs)
            return output

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Get the parameters of the model

        Args:
            recurse (bool, optional): Whether to get the parameters recursively. Defaults to True.

        Yields:
            Iterator[Parameter]: The iterator of the parameters
        """
        yield from self._model.parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Get the named parameters of the model
        """
        return super().named_parameters(prefix, recurse, remove_duplicate)


    """Utility functions"""

    @torch.no_grad()
    def global_avg(self):
        """Perform global average on the parameters (and buffers if sync_buffer_in_global_avg is True)
            The function is called at the end of the training loop to synchronize the parameters across all nodes for evaluation
        """
        self._over_communication() #处理之前的通信
        print('开始全局平均')
        bf.barrier()
        handles =[]
        if self._param_as_bucket_view:
            #此时同步模型桶参数
            for i in range(len(self._param_blocks)):
                handle = bf.allreduce_nonblocking_(self._param_blocks[i], average=False,name =f'_param_blocks{i}' )
                handles.append(handle)
            for handle in handles:
                bf.synchronize(handle)
        else: #直接同步参数
            bf.allreduce_parameters(self._params)
        
        if self._sync_buffer_in_global_avg:
                  
                  for name,x in self._model.named_buffers():
                       if x.dtype in self.FLOAT_DTYPES:
                         bf.allreduce_(x.data, average=False, name=f"bufavg.{name}")
                         x.data.div_(self._world_size)

        bf.barrier()
        

__all__ = ['DecentralizedDataParallel',
           'OPTIM_FN_TYPE',
           'LR_SCHEDULER_FN_TYPE',
           'LocalStepRandomBatchSampler']
