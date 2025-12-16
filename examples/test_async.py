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
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data.distributed
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
# parser.add_argument('--disable-dynamic-topology', action='store_true',
#                     default=False, help=('Disable each iteration to transmit one neighbor ' +
#                                          'per iteration dynamically.'))
# parser.add_argument('--atc-style', action='store_true', default=False,
#                     help='If True, the step of optimizer happened before communication')

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

# model = Net()
world_size = bf.size()
rank = bf.rank()

# topology,nx 的有向图digraph
topology = {'topology':topology_util.FullyConnectedGraph(world_size),'name': 'full'}



if args.cuda:
    print("using cuda.")
    # Move model to GPU.
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
# 写ddp的训练函数
optim_fn = partial(optim.optim_fn_adam, beta1=0.974, lr=1e-3 * bf.size())
model = ddp(model,
            optim_fn,
            lr_scheduler_fn=None,
            sync=False,
            topology=topology,
            sync_buffer_in_global_avg=True)

model.set_topology(topology)
print('ok')
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
# train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
# valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

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
# for epoch in range(200):
model.global_avg()
logger.debug('完成global_avg')
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

# bf.barrier()
# print('开始进入验证阶段',rank)
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
