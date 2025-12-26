Asyncddp
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

This project implements asynchronous Distributed Data Parallel (DDP) training. Follow the instructions below to set up your environment and install the package.

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

---

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

BlueFog is able to run interactively with Jupyter Notebook. Please check out our `hello world notebook <https://github.com/Bluefog-Lib/bluefog/blob/master/examples/interactive_bluefog_helloworld.ipynb>`_ or other notebooks in the example folder to start.

Interactive BlueFog is great for research and algorithmic experiments. For large-scale machine learning problems, we recommend using BlueFog with a script.

Using Asyncddp With Script
^^^^^^^^^^^^^^^^^^^^^^^^^
 For the dynamic topology case, you need a little more code:

.. code-block:: python

   from bluefog.common import topology_util
   # ...
   # Same setup code as previous snippets
   dynamic_neighbors_gen = topology_util.GetInnerOuterExpo2DynamicSendRecvRanks(
       bf.size(), local_size=bf.local_size(), self_rank=bf.rank())

   def dynamic_topology_update(epoch, batch_idx):
       send_neighbors, recv_neighbors = next(dynamic_neighbors_gen)
       avg_weight = 1/(len(recv_neighbors) + 1)
       optimizer.send_neighbors = to_neighbors
       optimizer.neighbor_weights = {r: avg_weight for r in recv_neighbors}
       optimizer.self_weight = avg_weight

   # Torch training code
   for epoch in range(epochs):
       for batch_idx, (data, target) in enumerate(train_loader):
           dynamic_topology_update(epoch, batch_idx)
           # ...
           loss.backward()
           optimizer.step()

Check our `dynamic topology neighbor averaging <https://bluefog-lib.github.io/bluefog/neighbor_average.html>`_ page to see more on how to control and use topology. See the BlueFog `examples`_ folder for full code.

We also provide many low-level functions which you can use as building blocks to construct your own distributed training algorithms. The following example illustrates how to run a simple consensus algorithm through Asyncddp.

.. code-block:: python

   import torch
   import bluefog.torch as bf

   bf.init()
   x = torch.Tensor([bf.rank()])
   for _ in range(100):
       x = bf.neighbor_allreduce(x)
   print(f"{bf.rank()}: Average value of all ranks is {x}")

Checkout our `API explanation page <https://bluefog-lib.github.io/bluefog/bluefog_ops.html>`_ to see all supported *synchronous* and *asynchronous* features.

The Asyncddp source code was based off the `Bluefog <https://github.com/Bluefog-Lib/bluefog.git>`_ repository. Hence, Asyncddp shares many common features with Bluefog such as `timeline <https://bluefog-lib.github.io/bluefog/timeline.html>`_, tensor-fusion, etc. We want to express our gratitude to the Bluefog team.

