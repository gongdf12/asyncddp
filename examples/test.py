import argparse
import torch
import bluefog.torch as bf
from bluefog.common import topology_util

def main():
    parser = argparse.ArgumentParser(description="Bluefog win_put() 测试")
    parser.add_argument("--dim", type=int, default=8, help="张量长度")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--cuda", action="store_true", help="使用 GPU（若可用）")
    parser.add_argument("--name", type=str, default="w", help="窗口名称")
    args = parser.parse_args()

    bf.init()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda", bf.local_rank() % torch.cuda.device_count()) if use_cuda else torch.device("cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # 建议使用 ring 拓扑，入/出邻居更可预测
    bf.set_topology(topology_util.RingGraph(bf.size(), connect_style=1))

    # 每个 rank 用常量（自己的 rank 编号）填充张量，方便校验
    x = torch.full((args.dim,), float(bf.rank()), dtype=dtype, device=device)

    # 为该张量创建窗口（邻居缓冲区 zero_init=True）
    ok = bf.win_create(x, name=args.name, zero_init=True)
    if not ok:
        raise RuntimeError(f"Rank {bf.rank()}: win_create 失败")

    # 准备向所有出邻居 put
    dst_weights = {r: 1.0 for r in bf.out_neighbor_ranks()}

    # 进行阻塞式 win_put（带互斥，避免与对端 update 并发）
    ok = bf.win_put(x, name=args.name, self_weight=1.0, dst_weights=dst_weights, require_mutex=True)
    if not ok:
        raise RuntimeError(f"Rank {bf.rank()}: win_put 失败")

    # 全局同步（非必须，但利于演示稳定性）
    bf.barrier()

    # 从入邻居收集数据到本地（仅邻居参与，self_weight=0.0；并在收集后重置邻居缓冲）
    in_ranks = bf.in_neighbor_ranks()
    neighbor_weights = {r: 1.0 for r in in_ranks}
    y = bf.win_update(
        name=args.name,
        self_weight=0.0,
        neighbor_weights=neighbor_weights,
        reset=True,
        require_mutex=True,
        clone=True,  # 返回新张量，避免覆盖窗口里注册的原张量引用
    )

    # 期望值：邻居张量均为常量“邻居 rank”，按权重 1.0 相加
    expected_val = sum(float(r) for r in in_ranks)
    expected = torch.full((args.dim,), expected_val, dtype=dtype, device=device)

    ok = torch.allclose(y, expected)
    print(
        f"[rank {bf.rank()}] in_neighbors={in_ranks}, "
        f"expected={expected_val}, ok={bool(ok)}"
    )
    if not ok:
        # 打印前几个元素便于调试
        print(f"[rank {bf.rank()}] y[:4]={y[:4].tolist()}, exp[:4]={expected[:4].tolist()}")

    # 清理
    bf.barrier()
    bf.win_free(args.name)

if __name__ == "__main__":
    main()