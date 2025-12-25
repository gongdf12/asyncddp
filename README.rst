Asyncddp
========

Asyncddp is a package designed for asynchronous model training. An example of asynchronous model training is provided in ``example/test_async-py``.

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

First, ensure your environment meets the following requirements: ``python>=3.7`` and ``openmpi >= 4.0``.

Then, install Bluefog.

**Standard Install:**

.. code-block:: shell

   pip install --no-cache-dir bluefog

**Install with NCCL Support:**
(If NCCL is supported, i.e., ``NCCL>=2.7``)

.. code-block:: shell

   BLUEFOG_WITH_NCCL=1 pip install bluefog

.. note::

   It should be noted that after installation, you must check if the header file is visible to the compiler. Use the following commands to confirm:

   .. code-block:: shell

      ls -l /usr/lib/libnccl*
      # OR
      ls -l /usr/local/nccl-<version>/lib/libnccl*

   Check the `install_bluefog <https://bluefog-lib.github.io/bluefog/install.html>`_ page if you need more information or other install options.

Using BlueFog With Jupyter Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BlueFog is able to run interactively with Jupyter Notebook. Please check out our `hello world notebook <https://github.com/Bluefog-Lib/bluefog/blob/master/examples/interactive_bluefog_helloworld.ipynb>`_ or other notebooks in the example folder to start.

Interactive BlueFog is great for research and algorithmic experiments. For large-scale machine learning problems, we recommend using BlueFog with a script.

Using BlueFog With Script
^^^^^^^^^^^^^^^^^^^^^^^^^

We provide a high-level wrapper for the torch optimizer. To convert an existing script to a distributed implementation, you simply need to wrap the optimizer with our ``DistributedNeighborAllreduceOptimizer`` and run it through ``bfrun``. That is it!

.. code-block:: python

   # Execute Python functions in parallel through:
   # bfrun -np 4 python file.py

   import torch
   import bluefog.torch as bf
   # ...
   bf.init()
   optimizer = optim.SGD(model.parameters(), lr=lr * bf.size())
   optimizer = bf.DistributedNeighborAllreduceOptimizer(
       optimizer, model=model
   )
   # ...

The previous example is for static topology usage. For the dynamic topology case, you need a little more code:

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

Check our BlueFog `dynamic topology neighbor averaging <https://bluefog-lib.github.io/bluefog/neighbor_average.html>`_ page to see more on how to control and use topology. See the BlueFog `examples`_ folder for full code.

We also provide many low-level functions which you can use as building blocks to construct your own distributed training algorithms. The following example illustrates how to run a simple consensus algorithm through Bluefog.

.. code-block:: python

   import torch
   import bluefog.torch as bf

   bf.init()
   x = torch.Tensor([bf.rank()])
   for _ in range(100):
       x = bf.neighbor_allreduce(x)
   print(f"{bf.rank()}: Average value of all ranks is {x}")

Checkout our `API explanation page <https://bluefog-lib.github.io/bluefog/bluefog_ops.html>`_ to see all supported *synchronous* and *asynchronous* features.

The Bluefog source code was based off the `Horovod <https://github.com/horovod/horovod>`_ repository. Hence, BlueFog shares many common features with Horovod such as `timeline <https://bluefog-lib.github.io/bluefog/timeline.html>`_, tensor-fusion, etc. We want to express our gratitude to the Horovod team.

Materials
---------

*Bluefog: Make decentralized algorithms practical for optimization and deep learning*. B. Ying, K. Yuan, H. Hu, Y. Chen, and W. Yin. arXiv preprint arXiv:2111.04287, 2021. `[link] <https://arxiv.org/abs/2111.04287>`_

*Faster Learning over Networks and BlueFog*, BlueFog Team, invited talk at MLA, 2020 `[slides] <https://github.com/Bluefog-Lib/bluefog/blob/master/resources/Faster_Learning_over_Networks_and_BlueFog.pdf>`_

Cite
----

Bluefog is uploaded to Zenodo. An equivalent BibTex format reference is below for all versions:

.. code-block:: bibtex

     % System paper
     @article{bluefog,
       author        = {Ying, Bicheng and Yuan, Kun and Hu, Hanbin and Chen, Yiming and Yin, Wotao},
       title         = {BlueFog: Make Decentralized Algorithms Practical for Optimization and Deep Learning},
       journal       = {arXiv preprint arXiv:2111.04287},
       year          = {2021},
     }

     % Theoretical Papers
     @article{ying2021exponential,
       title={Exponential Graph is Provably Efficient for Decentralized Deep Training},
       author={Ying, Bicheng and Yuan, Kun and Chen, Yiming and Hu, Hanbin and Pan, Pan and Yin, Wotao},
       journal={Advances in Neural Information Processing Systems (NeurIPS), 34.
                Also available at arXiv:2110.13363},
       year={2021}
     }

     @inproceedings{yuan2021decentlam,
        title={DecentLaM: Decentralized Momentum SGD for Large-Batch Deep Training},
        author={Yuan, Kun and Chen, Yiming and Huang, Xinmeng and Zhang, Yingya and Pan, Pan and Xu, Yinghui and Yin, Wotao},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={3029--3039},
        year={2021}
     }

     @article{yuan2020influence,
        title={On the influence of bias-correction on distributed stochastic optimization},
        author={Yuan, Kun and Alghunaim, Sulaiman A and Ying, Bicheng and Sayed, Ali H},
        journal={IEEE Transactions on Signal Processing},
        volume={68},
        pages={4352--4367},
        year={2020},
        publisher={IEEE}
     }

Troubleshooting
---------------

Import bluefog.torch failed
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see the error message below, it means that bluefog is not installed properly. Please install bluefog using the github source and recompile bluefog (e.g. ``make clean && make -j $(nproc) && BLUEFOG_WITH_NCCL=1 pip install .``).

.. code-block:: python

    import bluefog.torch as bf
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/usr/local/lib/python3.7/dist-packages/bluefog/torch/__init__.py", line 34, in <module>
        from bluefog.torch.mpi_ops import init, shutdown
    File "/usr/local/lib/python3.7/dist-packages/bluefog/torch/mpi_ops.py", line 23, in <module>
        from bluefog.torch import mpi_lib  # C library
    ImportError: /usr/local/lib/python3.7/dist-packages/bluefog/torch/mpi_lib.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZN7bluefog6common14NCCLController9AllreduceERNS0_16TensorTableEntryE

.. _AWS: https://aws.amazon.com/about-aws/whats-new/2018/12/introducing-amazon-ec2-p3dn-instances-our-most-powerful-gpu-instance-yet/
.. _examples: https://github.com/Bluefog-Lib/bluefog/tree/master/examples
