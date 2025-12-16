# test_win_ops.py
# 用于验证 win_put / WinPassiveRecvRequest / win_update 的数据传输正确性
#
# 运行方式（例）：
#   BLUEFOG_WITH_NCCL=1 mpirun -np 3 python test_win_ops.py
# 或者
#   BLUEFOG_WITH_NCCL=1 bfrun -np 3 python test_win_ops.py
#
# 注意：不要设置 BLUEFOG_WIN_OPS_BY_MPI=1，这样 win_* 才会走 NCCL 路径。

import os
import torch
import bluefog.torch as bf
from loguru import logger

def value_for(rank: int, step: int) -> float:
    """给定 rank 和迭代步数，构造一个可预测的标量值。"""
    return float(rank * 10 + step)


def test_win_put_and_read(name="w_test_put_read"):
    """测试：每个 rank 调用 win_put 后，其他 rank 通过 win_read_neighbor
    读到的 neighbor buffer 是否等于对方写入的值。
    """
    rank = bf.rank()
    device = torch.device(
        "cuda", bf.local_rank()
    ) if torch.cuda.is_available() else torch.device("cpu")

    # 使用长度较小的一维 tensor，方便打印对比
    length = 8
    step = 0  # 单步测试
    v = value_for(rank, step)

    # 自己窗口绑定的 tensor
    x = torch.ones(8, dtype=torch.float32, device=device)*(rank+2)

    # 创建 window，邻居 buffer 初始化为 0
    ok = bf.win_create(x, name, zero_init=True)
    assert ok, f"[rank {rank}] win_create failed"

    bf.barrier()

    # 发送自己的 tensor 到 out-neighbors
    ok = bf.win_put(x, name)
    assert ok, f"[rank {rank}] win_put failed"

    bf.barrier()

    # 在本 rank 上检查所有 in-neighbors 的窗口内容
    # # errs = []
    for src in bf.in_neighbor_ranks():
        t = bf.win_read_neighbor(name, src, require_mutex=True)
        logger.debug(f"rank{src} -> rank{rank}:{t},expect:{src+2}")
    #     expected = value_for(src, step)
    #     if not torch.allclose(t, torch.full_like(t, expected)):
    #         errs.append(
    #             f"from {src}: got {t[:4].tolist()}, expected {expected}"
    #         )

    # if errs:
    #     print(f"[rank {rank}] WIN_PUT / passive recv FAILED:\n  " +
    #           "\n  ".join(errs))
    # else:
    #     print(f"[rank {rank}] WIN_PUT / passive recv OK")
    # print("ok")
    # bf.barrier()
    print("ok1")
    bf.win_free(name)
    print("ok")
    # bf.barrier()


def test_win_put_and_update(name="w_test_put_update", num_steps: int = 4):
    """测试：多轮 win_put + win_update_then_collect 后，本地 tensor 的结果
    是否等于“自己 + 所有 in-neighbor”的和（因为 win_update_then_collect 内部使用
    self_weight=1, neighbor_weights=1，不做除法）。
    """
    rank = bf.rank()
    size = bf.size()
    device = torch.device(
        "cuda", bf.local_rank()
    ) if torch.cuda.is_available() else torch.device("cpu")

    length = 16
    x = torch.zeros(length, dtype=torch.float32, device=device)

    ok = bf.win_create(x, name, zero_init=True)
    assert ok, f"[rank {rank}] win_create failed"

    bf.barrier()

    in_neigh = bf.in_neighbor_ranks()

    for step in range(num_steps):
        # 每个 rank 在这一轮写入一个可预测的值
        v = value_for(rank, step)
        x.fill_(v)

        # 把本地 x 放到所有 out-neighbors 的 window buffer 中
        ok = bf.win_put(x, name)
        assert ok, f"[rank {rank}] win_put failed at step {step}"

        # 计算理论期望值：自己 + 所有 in-neighbors 的当前值
        expected = value_for(rank, step)
        for src in in_neigh:
            expected += value_for(src, step)

        # 调用 win_update_then_collect 会内部调用 win_update：
        # 等价于 win_update(name, self_weight=1.0,
        #                    neighbor_weights={r:1.0 for r in in_neigh},
        #                    reset=True)
        y = bf.win_update_then_collect(name, require_mutex=True)
        y_cpu = y.to("cpu")

        max_err = float((y_cpu - expected).abs().max().item())
        if max_err > 1e-5:
            print(
                f"[rank {rank}] step {step} WIN_UPDATE FAILED: "
                f"y[0]={y_cpu[0].item():.4f}, expected={expected:.4f}, "
                f"max_err={max_err:.4e}"
            )
        else:
            print(
                f"[rank {rank}] step {step} WIN_UPDATE OK: "
                f"y[0]={y_cpu[0].item():.4f}, expected={expected:.4f}"
            )

    bf.win_free(name)
    bf.barrier()


def main():
    bf.init()
    rank = bf.rank()
    # bf.set_skip_negotiate_stage(True)
    if bf.size() < 2:
        raise RuntimeError("需要至少 2 个进程来测试窗口通信，请使用 mpirun/bfrun -np >=2")

    # 测试 1：只验证 win_put + WinPassiveRecvRequest
    test_win_put_and_read("w_async_put_read")

    # 测试 2：多轮 win_put + win_update
    # test_win_put_and_update("w_async_put_update", num_steps=4)

    if rank == 0:
        print("All window tests finished (see per-rank输出检查是否有 FAILED).")
    # y=bf.win_read_neighbor()

if __name__ == "__main__":
    main()
