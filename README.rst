AsyncDDP
========

Asyncddp is a package designed for asynchronous model training. An example of asynchronous model training is provided in ``example/test_async-py``.
When running these files, it is necessary to modify the machine parameters in the script.

Manual
------

To understand the details of the communication functions, please checkout the `performance page <https://bluefog-lib.github.io/bluefog/performance.html>`_.

Overview
--------

Asyncddp is built upon decentralized optimization algorithms. This is fundamentally different from other popular distributed training frameworks, such as DistributedDataParallel provided by PyTorch, Horovod, BytePS, etc.

In each communication stage, neither the typical star-shaped parameter-server topology nor the pipelined ring-allreduce topology is used. Instead, BlueFog exploits a virtual and potentially dynamic network topology (which can take any shape) to achieve maximum communication efficiency.

**Main Idea: Replace expensive allreduce averaging over gradients with cheap neighbor averaging over parameters.**

For each training iteration, one process (or agent) will update its model using information received from its **direct** neighbors as defined by the virtual topology. It is observed that all communication occurs only over the predefined virtual topology, and no global communication is required. This is why the algorithm is named *decentralized*. Decentralized training algorithms have been proven in literature to converge to the same solution as their standard centralized counterparts.

The topology determines communication efficiency. BlueFog supports both **static** topology and **dynamic** topology usage. After extensive trials, the dynamic Exponential-2 graph was observed to achieve the best performance if the number of agents is a power of 2 (e.g., 4, 32, 128 agents). In an Exponential-2 graph, each agent communicates with neighbors that are :math:`2^0, 2^1, ..., 2^t` hops away. **Dynamic** topology means all agents select only one neighbor in one iteration and select the next neighbor in the next iteration, as illustrated in the following figure:

.. raw:: html

    <p align="center"><img src="https://user-images.githubusercontent.com/16711681/97928035-04654400-1d1b-11eb-91d2-2da890b4522e.png" alt="one-peer-exp2" width="650"/></p>

In this scenario, the communication cost for each iteration is only one unit of delay and one standard parameter size to transmit. No communication conflicts occur, which is superior to the guarantees provided by parameter server or ring-allreduce methods.

Quick Start
-----------

# AsyncDDP Installation

This project implements Asynchronous Distributed Data Parallel (AsyncDDP) training. Follow the instructions below to set up your environment and install the package.

## 🛠 Prerequisites

Before installation, ensure your system meets the following hardware and software requirements:

* **OS**: Linux (Recommended for distributed training)
* **Hardware**: NVIDIA GPU + CUDA (Required for DDP)
* **Package Manager**: Conda (Anaconda or Miniconda)

### Environment Dependencies
The following versions are strictly required:

| Dependency   | Version Requirement |
| :----------- | :------------------ |
| **Python** | `>= 3.12.7`         |
| **OpenMPI** | `>= 4.0`            |
| **NCCL** | `== 2.28.9`         |
| **Flatbuffers** | `1.12.0`          |
| **Boost** | `>= 1.74.0`         |
| **PyTorch** | Latest compatible   |

**⚠️ Notice**
Please use pip to install the complete Pytorch from the official website


## 🚀 Installation Steps

We provide an automated script to handle environment creation and dependency installation.


```bash
git clone https://github.com/gongdf12/asyncddp.git
cd asyncddp
# Grant execute permission
#chmod +x set_conda_down.sh
# Run the setup script
bash set_conda_down.sh```
  
Using Asyncddp With Jupyter Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BlueFog is able to run interactively with Jupyter Notebook. Please check out  `hello world notebook <https://github.com/Bluefog-Lib/bluefog/blob/master/examples/interactive_bluefog_helloworld.ipynb>`_ or other notebooks in the example folder to start.

Interactive BlueFog is great for research and algorithmic experiments. For large-scale machine learning problems, we recommend using BlueFog with a script.

Using Asyncddp to train a model
^^^^^^^^^^^^^^^^^^^^^^^^^
 For the asynchronous training model , you need a little more code:

.. code-block:: python

            from __future__ import print_function
            
            from bluefog.common import topology_util
            import bluefog.torch as bf
            import torch
            import argparse
            import os
            import sys
            import warnings
            warnings.simplefilter('ignore')
            from tqdm import tqdm
            from loguru import logger
            import time
            # from torchvision import datasets, transforms
            from bluefog.torch.ddp import DecentralizedDataParallel as ddp
            from bluefog.torch.ddp import LocalStepRandomBatchSampler as LRsampler
            import bluefog.torch.optim as optim
            from functools import partial
            import torchvision
            from torch.utils.data import DistributedSampler, DataLoader
            sys.path.insert(0, os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")))
            
            # Training settings
            parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
            parser.add_argument(
                "--batch-size", type=int, default=32,
                metavar="N", help="input batch size for training (default: 64)")
            parser.add_argument(
                "--test-batch-size", type=int, default=32,
                metavar="N", help="input batch size for testing (default: 1000)")
            parser.add_argument("--epochs", type=int, default=10, metavar="N",
                                help="number of epochs to train (default: 10)")
            parser.add_argument(
                "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")
            parser.add_argument("--momentum", type=float, default=0.5,
                                metavar="M", help="SGD momentum (default: 0.5)")
            parser.add_argument(
                "--no-cuda", action="store_true", default=False, help="disables CUDA training")
            parser.add_argument('--dist-optimizer', type=str, default='neighbor_allreduce',
                                help='The type of distributed optimizer. Supporting options are ' +
                                '[neighbor_allreduce, hierarchical_neighbor_allreduce, allreduce, horovod]')
            
            parser.add_argument(
                "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
            )
            parser.add_argument(
                "--log-interval",
                type=int,
                default=10,
                metavar="N",
                help="how many batches to wait before logging training status",
            )
            
            args = parser.parse_args()
            args.cuda = not args.no_cuda and torch.cuda.is_available()
            torch.manual_seed(args.seed)
            
            if args.dist_optimizer == 'horovod':
                print("importing horovod")
                import horovod.torch as bf
            
            bf.init()
            # bf.set_skip_negotiate_stage(True)
            if args.cuda:
                # Bluefog: pin GPU to local rank.
                device_id = bf.local_rank() if bf.nccl_built() else bf.local_rank() % torch.cuda.device_count()
                torch.cuda.set_device(device_id)
                torch.cuda.manual_seed(args.seed)
            
            
            kwargs = {"num_workers": 0, "pin_memory": False} if args.cuda else {}
            
            
            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(784, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10),
            )
            
            world_size = bf.size()
            rank = bf.rank()
            topology = {'topology':topology_util.FullyConnectedGraph(world_size),'name': 'full'}
            if args.cuda:
                print("using cuda.")
                # Move model to GPU.
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
           # 可以自定义优化器
            optim_fn = partial(optim.optim_fn_adam, beta1=0.974, lr=1e-3 * bf.size())
            model = ddp(model,
                        optim_fn,
                        lr_scheduler_fn=None,
                        sync=False,
                        topology=topology,
                        sync_buffer_in_global_avg=True)
            
            model.set_topology(topology)
            train_dataset = torchvision.datasets.MNIST(
                train=True,
                download=True,
                root='.',
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ])
            )
            valid_dataset = torchvision.datasets.MNIST(
                train=False,
                download=True,
                root='.',
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ])
            )
            
            # 关于样本批次的抽样规则可以在ddp中自定义，这个地方只是示例
            train_sampler = LRsampler(train_dataset,
                                      batch_size=256 // world_size+20*rank ,
                                      base_seed=42+rank,
                                      rank=rank,
                                      drop_last=True) 
            valid_sampler = LRsampler(valid_dataset,
                                      batch_size=256 // world_size+20*rank,
                                      base_seed=42 +rank,
                                      rank=rank,
                                      drop_last=True)
            
            train_ds = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  pin_memory=False,
                                  )
            valid_ds = DataLoader(valid_dataset,
                                   batch_sampler=valid_sampler,
                                   pin_memory=False,
                                   )
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.025)
            
            print('start training')
            model.global_avg()
            max_seconds = 30 
            start_time = time.time()
            epoch = 0
            model.train()
            while True:
                    if time.time() - start_time >= max_seconds:
                        break
            
                    # 训练阶段（本地 step 和随机性由 LocalStepRandomBatchSampler 控制）
                    # model.train()
                    train_loss = 0.0
                    train_acc = 0.0
                    num_train_batches = 0
            
                    with tqdm(train_ds, desc=f"[Rank {rank}] Epoch {epoch} [Train]") as t:
                        for data, target in t:
                
                              data = data.to(device, non_blocking=True)
                              target = target.to(device, non_blocking=True)
                              output = model(data)
                              loss = loss_fn(output, target)
                              loss.backward()
                              batch_acc = (output.argmax(1) == target).float().mean().item()
                              train_loss += loss.item()
                              train_acc += batch_acc
                              num_train_batches += 1
            
                              t.set_postfix({
                              "loss": f"{train_loss / num_train_batches:.4f}",
                              "acc": f"{train_acc / num_train_batches:.4f}",
                              })
                                # print(time.time()-start_time,'---------------------------------')
                    if num_train_batches > 0 :
                        print(f"[Train] loss={train_loss / num_train_batches:.4f}, "
                              f"acc={train_acc / num_train_batches:.4f}")
                    print(time.time()-start_time,'---------------------------------')
                    if time.time() - start_time >= max_seconds:
                        break
        
            model.global_avg()
            model.eval()
            valid_loss = 0.0
            valid_acc = 0.0
            num_valid_batches = 0
            with torch.no_grad():
                for data, target in valid_ds:
                    if time.time() - start_time >= max_seconds:
                        break
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
            
                    output = model(data)
                    loss = loss_fn(output, target)
                    batch_acc = (output.argmax(1) == target).float().mean().item()
            
                    valid_loss += loss.item()
                    valid_acc += batch_acc
                    num_valid_batches += 1
            
            if num_valid_batches > 0 and rank == 0:
                print(f"[Valid] loss={valid_loss / num_valid_batches:.4f}, "
                        f"acc={valid_acc / num_valid_batches:.4f}")
            epoch += 1

Check  `dynamic topology neighbor averaging <https://bluefog-lib.github.io/bluefog/neighbor_average.html>`_ page to see more on how to control and use topology. See the BlueFog `examples`_ folder for full code.

We also provide many low-level functions which you can use as building blocks to construct your own distributed training algorithms. The following example illustrates how to run a simple consensus algorithm through Asyncddp.

.. code-block:: python

   import torch
   import bluefog.torch as bf

   bf.init()
   x = torch.Tensor([bf.rank()])
   for _ in range(100):
       x = bf.neighbor_allreduce(x)
   print(f"{bf.rank()}: Average value of all ranks is {x}")

Checkout  `API explanation page <https://bluefog-lib.github.io/bluefog/bluefog_ops.html>`_ to see all supported *synchronous* and *asynchronous* features.

## 🙏 Acknowledgments

The AsyncDDP source code is built upon the following open-source projects:

- **[Bluefog](https://github.com/Bluefog-Lib/bluefog)** - A high-performance communication library for distributed training.
- **[Decent-DP](https://github.com/WangZesen/Decent-DP)** - A research project on decentralized data parallelism.

We extend our sincere gratitude to the original authors and contributors of these projects.
