"""
Bluefog window async communication test.

- Creates a window for a per-rank tensor (filled with rank id)
- Posts nonblocking window ops (win_put or win_accumulate) to out-neighbors
- Overlaps with dummy compute, then waits and collects via win_update
- Validates the received tensor equals expected neighbor aggregation

Run examples:
  CPU: bfrun -np 2 python examples/test_win_async.py
  GPU: CUDA_VISIBLE_DEVICES=0,1 bfrun -np 2 --network-interface eno1 \
       -x BLUEFOG_WIN_ON_GPU=1 python examples/test_win_async.py --cuda
"""
import argparse
import time
import torch
import bluefog.torch as bf
from bluefog.common import topology_util

import os, sys

# os.makedirs("logs", exist_ok=True)
# r = bf.rank()
# log_path = os.path.join("logs", f"rank_{r}.log")

# # 把当前 rank 的 stdout/stderr 重定向到自己的文件
# log_f = open(log_path, "w", buffering=1)
# sys.stdout = log_f
# sys.stderr = log_f

def dummy_compute(device: torch.device, iters: int = 1000) -> None:
    # Small matmul to occupy some time and show overlap potential.
    a = torch.randn(64, 64, device=device)
    b = torch.randn(64, 64, device=device)
    c = a
    for _ in range(iters):
        c = c @ b
    torch.cuda.synchronize() if device.type == 'cuda' else None


def main() -> None:
    ap = argparse.ArgumentParser("Bluefog win async test")
    ap.add_argument("--dim", type=int, default=8, help="tensor length")
    ap.add_argument("--iters", type=int, default=3, help="iterations")
    ap.add_argument("--op", choices=["put", "accumulate"], default="put",
                    help="which async win op to test")
    ap.add_argument("--cuda", action="store_true", help="use CUDA if available")
    ap.add_argument("--name", type=str, default="w_async", help="window name")
    ap.add_argument("--topo", type=str, default="full",
                    choices=["ring", "expo2", "full"], help="virtual topology")
    ap.add_argument("--compute-iters", type=int, default=1000,
                    help="dummy compute iterations to overlap")
    ap.add_argument("--mode", type=str, default="single",
                    choices=["single", "pingpong"], help="async pattern")
    args = ap.parse_args()

    bf.init()
    bf.set_skip_negotiate_stage(True)
    rank = bf.rank()
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(bf.local_rank() % torch.cuda.device_count())
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Configure a simple and predictable topology.
    if args.topo == "ring":
        bf.set_topology(topology_util.RingGraph(bf.size(), connect_style=1))
    elif args.topo == "expo2":
        bf.set_topology(topology_util.ExponentialGraph(bf.size(), base=2))
    else:  # full
        bf.set_topology(topology_util.FullyConnectedGraph(bf.size()))
    G = topology_util.RingGraph(bf.size(), connect_style=1)
    # print(type(G))
    # Each rank publishes an initial vector to define the window shape.
    # Actual payloads are computed per-iteration below.
    x = torch.full((args.dim,), float(bf.rank()), dtype=torch.float32, device=device)
    # print(x)
    ok = bf.win_create(x, name=args.name, zero_init=True)
    if not ok:
        raise RuntimeError(f"Rank {bf.rank()}: win_create failed")

    in_ranks = bf.in_neighbor_ranks()
    out_ranks = bf.out_neighbor_ranks()
   

    #测试allreduce 函数
    # bf.allreduce_(x,average=True,name= args.name)
    # print(x)
    # bf.barrier()
    for it in range(args.iters): # 开始测试每一轮迭代换一次topo
        bf.set_skip_negotiate_stage(True)
        payload = torch.full((args.dim,), float(bf.rank() + it),
                                dtype=torch.float32, device=device)
        send_buf = payload.clone()

        dst_weights = {r: 1.0 for r in out_ranks}
        if args.op == "put":
            print('进入新一轮通信',it,rank)
            if rank==0:  # 模拟时间结束的情况
                pass
            else:
                handle = bf.win_put_nonblocking(
                    send_buf, name=args.name, self_weight=1.0, dst_weights=dst_weights, require_mutex=True
                )
        else:
            handle = bf.win_accumulate_nonblocking(
                send_buf, name=args.name, self_weight=1.0, dst_weights=dst_weights, require_mutex=True
            )

        if args.compute_iters > 0:
            dummy_compute(device, iters=args.compute_iters)
            time.sleep(5*bf.rank())

        # bf.win_wait(handle)
        # handlelist.append(handle)
        neighbor_weights = {r: 1.0 for r in in_ranks}
        # for r in bf.in_neighbor_ranks():
        #         buf = bf.win_read_neighbor(args.name, r, require_mutex=True)
        #         # 这里只做简单打印或断言
        #         print(f"[rank {bf.rank()}] read from {r}, buf[0]={buf.view(-1)[0].item()}")
        y = bf.win_update(
            name=args.name,
            self_weight=0.0,
            neighbor_weights=neighbor_weights,
            reset=True,
            require_mutex=True,
            clone=True,
        )
        
        expected_val = sum(float(r + it) for r in in_ranks)
        expected = torch.full((args.dim,), expected_val, dtype=torch.float32, device=device)
        ok = torch.allclose(y, expected, atol=1e-5)
        print(
            f"[rank {bf.rank()}] iter={it} op={args.op} in={in_ranks} out={out_ranks} ok={bool(ok)}",
            flush=True,
        )
        if not ok:
            print(f"[rank {bf.rank()}] y[:4]={y[:4].tolist()} exp[:4]={expected[:4].tolist()}", flush=True)
        # bf.barrier()
        # handle=bf.allreduce_nonblocking_(y,average=True,name= args.name)
        # bf.synchronize(handle)
    bf.win_free(args.name)


if __name__ == "__main__":
    main()
