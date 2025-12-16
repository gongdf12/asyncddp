import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Any, List, Tuple, Union


def _get_param_groups(params: List[Tuple[str, Tensor]], weight_decay: float) -> list:
    """Get the parameters grouped by weight decay and no weight decay.
     #weight_decay 是步长
    Returns:
        dict: a dictionary with two keys, 'params' and 'params_no_decay'
    """
    params_no_decay = [x for n, x in params if not (('bn' in n) or ('bias' in n))]
    params_decay = [x for n, x in params if ('bn' in n) or ('bias' in n)]
    #weight_decay 是步长,paramas 是张量列表
    return [
        {'params': params_no_decay, 'weight_decay': 0.},
        {'params': params_decay, 'weight_decay': weight_decay}
    ]
"-------------------下述三个函数展示传入参数产生优化器-------------------------------------------"

def optim_fn_adam(params: List[Tuple[str, Tensor]],
                  lr: float = 1e-3,
                  beta1: float = 0.9,
                  beta2: float = 0.999,
                  weight_decay: float = 1. / 32768,
                  eps: float = 1e-8) -> Optimizer:
    """An example of a function that creates an Adam optimizer with the given parameters and their names.
        To change the hyperparameters of the optimizer, you can wrap it with `functools.partial` and pass the new values.

    Returns:
        Optimizer: an Adam optimizer
    """
    return torch.optim.Adam(_get_param_groups(params, weight_decay), #这个函数返回一个列表作为paras,它只会作第一个参数传入
                            lr=lr,
                            betas=(beta1, beta2),
                            eps=eps)


def optim_fn_adamw(params: List[Tuple[str, Tensor]],
                   lr: float = 1e-3,
                   beta1: float = 0.9,
                   beta2: float = 0.999,
                   weight_decay: float = 0.1,
                   eps: float = 1e-8) -> Optimizer:
    """An example of a function that creates an AdamW optimizer with the given parameters and their names.
        To change the hyperparameters of the optimizer, you can wrap it with `functools.partial` and pass the new values.

    Returns:
        Optimizer: an AdamW optimizer
    """
    return torch.optim.AdamW(_get_param_groups(params, weight_decay),
                             lr=lr,
                             betas=(beta1, beta2),
                             eps=eps)
'------------------------------------------------------------------------------------------------------------------'


'---------------------------------使用accumadam类创建优化器-----------------------------------------------------------'
def optim_fn_accum_adam(params: List[Tuple[str, Tensor]],
                        lr: float = 1e-3,
                        beta1: float = 0.9,
                        beta2: float = 0.999,
                        eps: float = 1e-8,
                        weight_decay: float = 1. / 32768,
                        accum_iter: int = 4) -> Optimizer:
    """An example of a function that creates an AccumAdam optimizer with the given parameters and their names.
        To change the hyperparameters of the optimizer, you can wrap it with `functools.partial` and pass the new values.

    Returns:
        Optimizer: an AccumAdam optimizer
    """
    return AccumAdam(_get_param_groups(params, weight_decay),
                     lr=lr,
                     betas=(beta1, beta2),
                     eps=eps,
                     accum_iter=accum_iter)


def optim_fn_accum_adamw(params: List[Tuple[str, Tensor]],
                         lr: float = 1e-3,
                         beta1: float = 0.9,
                         beta2: float = 0.999,
                         eps: float = 1e-8,
                         weight_decay: float = 0.1,
                         accum_iter: int = 4) -> Optimizer:
    """An example of a function that creates an AccumAdamW optimizer with the given parameters and their names.
        To change the hyperparameters of the optimizer, you can wrap it with `functools.partial` and pass the new values.

    Returns:
        Optimizer: an AccumAdamW optimizer
    """
    return AccumAdamW(_get_param_groups(params, weight_decay),
                      lr=lr,
                      betas=(beta1, beta2),
                      eps=eps,
                      accum_iter=accum_iter)
'-------------------------------------------------------------------------------------------------------------------'



'-----------------------给优化器设置余弦学习率------------------------------------------------------------------------'

def lr_scheduler_fn_cosine_with_warmup(optimizer: Optimizer,
                                       t_max: int,
                                       t_warmup: int,
                                       cosine_eta_min: float = 1e-6,
                                       warmup_decay: float = 0.01) -> LRScheduler:
    """An example of a function that creates a learning rate scheduler that combines a warmup and a cosine annealing schedule.

    Returns:
        LRScheduler: a learning rate scheduler with the linear warmup followed by the cosine annealing
    """
    # T_max是周期，eta_min学习率下限
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=cosine_eta_min)
    #线性学习率预热，start_factor	float	初始学习率乘法因子（相对于初始学习率）total_iters	int	5	线性调整的总迭代次数
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_decay, total_iters=t_warmup)
    #顺序调度 各类学习率调度方法
    return torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                 schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                                                 milestones=[t_warmup])

'----------------------------关于累计adamw算法的实现--------------------------------------------'
def accum_adamw_foreach(params: List[torch.Tensor],
                       grads: List[torch.Tensor],
                       exp_avgs: List[torch.Tensor],
                       exp_avg_sqs: List[torch.Tensor],
                       accum_grads: List[torch.Tensor],
                       state_steps: List[torch.Tensor],
                       beta1: float,
                       beta2: float,
                       lr: Union[float, torch.Tensor],
                       weight_decay: float,
                       eps: float,
                       accum_iter: int):
    """Optimized version of AccumAdamW optimizer using torch._foreach
        TODO: fused kernel
    """
  # torch._foreach是对张量进行批量操作
    torch._foreach_add_(state_steps, 1)  #对列表内的张量做一个加法操作
    if weight_decay != 0:
        torch._foreach_mul_(params, 1 - lr * weight_decay) #全体张量乘一个标量

    step = state_steps[0].item()
    torch._foreach_add_(accum_grads, grads, alpha=1./accum_iter)  # ten1+=ten2*alpha

    _exp_avgs = torch._foreach_add(exp_avgs, grads, alpha=1-beta1) #同上
    _exp_avg_sqs = torch._foreach_addcmul(exp_avg_sqs, grads, grads, value=1-beta2) #t1+=a*(t2*t3)

    bias_correction1 = 1 - beta1 ** ((step + accum_iter - 1) // accum_iter)
    # // 取整数商 %取余 此处公式是在计算adam一阶动量的偏差修正系数
    bias_correction2 = 1 - beta2 ** ((step + accum_iter - 1) // accum_iter)
    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)

    torch._foreach_sqrt_(_exp_avg_sqs)
    torch._foreach_div_(_exp_avg_sqs, bias_correction2_sqrt) # 张量除法
    torch._foreach_add_(_exp_avg_sqs, eps)
    torch._foreach_addcdiv_(params, _exp_avgs, _exp_avg_sqs, value=-step_size) # type: ignore

    if step % accum_iter == 0:
        torch._foreach_add_(exp_avgs, accum_grads, alpha=1-beta1)
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_addcmul_(exp_avg_sqs, accum_grads, accum_grads, value=1-beta2)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_zero_(accum_grads)
      #张量的累计操作
    
"-------------------------------------------------------------------------------------------------"
class AccumAdamW(torch.optim.Optimizer):
    """AccumAdamW optimizer

    Args:
        params (Any): parameters list or groups
        lr (float, optional): base learning rate. Defaults to 1e-3.
        betas (Tuple[float, float], optional): beta1 and beta2. Defaults to (0.9, 0.999).
        eps (float, optional): epsilon. Defaults to 1e-8.
        weight_decay (float, optional): weight decay. Defaults to 0.
        accum_iter (int, optional): number of accumulation steps. Defaults to 4. should be scaling up with the number of workers.
    """
    
    def __init__(self,
                 params: Any,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 accum_iter: int = 4):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            accum_iter=accum_iter
        )
        super().__init__(params, defaults)

    def _init_group(self,
                    group,
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    accum_grads,
                    state_steps):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)  #存储梯度的列表
                grads.append(p.grad)
                state = self.state[p] #一个自动补全的字典。默认类型是tensor
                if len(state) == 0:
                    state['step'] = torch.tensor(0, dtype=torch.int64)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['accum_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 创建一个与现有张量形状相同，但是全为零的
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                accum_grads.append(state['accum_grad'])
                state_steps.append(state['step'])

    @torch.no_grad() #锁梯度
    def step(self, closure=None): # type: ignore
        self._cuda_graph_capture_health_check()
        assert closure is None

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            accum_grads = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_avg_sqs, accum_grads, state_steps
            ) #初始化相应参数为accum_adamw_foreach

            if len(state_steps) == 0:
                continue

            accum_adamw_foreach(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                accum_grads,
                state_steps,
                beta1,
                beta2,
                group['lr'],
                group['weight_decay'],
                group['eps'],
                group['accum_iter']
            )


def accum_adam_foreach(params: List[torch.Tensor],
                       grads: List[torch.Tensor],
                       exp_avgs: List[torch.Tensor],
                       exp_avg_sqs: List[torch.Tensor],
                       accum_grads: List[torch.Tensor],
                       state_steps: List[torch.Tensor],
                       beta1: float,
                       beta2: float,
                       lr: Union[float, torch.Tensor],
                       weight_decay: float,
                       eps: float,
                       accum_iter: int):
    """Optimized version of AccumAdam optimizer using torch._foreach
        TODO: write a fused kernel for this
    """
    torch._foreach_add_(state_steps, 1)
    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    step = state_steps[0].item()
    torch._foreach_add_(accum_grads, grads, alpha=1./accum_iter)

    _exp_avgs = torch._foreach_add(exp_avgs, grads, alpha=1-beta1)
    _exp_avg_sqs = torch._foreach_addcmul(exp_avg_sqs, grads, grads, value=1-beta2)

    bias_correction1 = 1 - beta1 ** ((step + accum_iter - 1) // accum_iter)
    bias_correction2 = 1 - beta2 ** ((step + accum_iter - 1) // accum_iter)
    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)

    torch._foreach_sqrt_(_exp_avg_sqs)
    torch._foreach_div_(_exp_avg_sqs, bias_correction2_sqrt)
    torch._foreach_add_(_exp_avg_sqs, eps)
    torch._foreach_addcdiv_(params, _exp_avgs, _exp_avg_sqs, value=-step_size) # type: ignore

    if step % accum_iter == 0:
        torch._foreach_add_(exp_avgs, accum_grads, alpha=1-beta1)
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_addcmul_(exp_avg_sqs, accum_grads, accum_grads, value=1-beta2)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_zero_(accum_grads)

'-----------------------------------------------------------------------------------------------------------------'
class AccumAdam(torch.optim.Optimizer):
    """AccumAdamW optimizer

    Args:
        params (Any): parameters list or groups
        lr (float, optional): base learning rate. Defaults to 1e-3.
        betas (Tuple[float, float], optional): beta1 and beta2. Defaults to (0.9, 0.999).
        eps (float, optional): epsilon. Defaults to 1e-8.
        weight_decay (float, optional): weight decay. Defaults to 0.
        accum_iter (int, optional): number of accumulation steps. Defaults to 4. should be scaling up with the number of workers.
    """
    def __init__(self,
                 params: Any,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1. / 32768,
                 accum_iter: int = 4):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            accum_iter=accum_iter
        )
        super().__init__(params, defaults)
        #继承optimizer包，初始化时必须有default字典和params

    def _init_group(self,
                    group,
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    accum_grads,
                    state_steps):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0, dtype=torch.int64)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['accum_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                accum_grads.append(state['accum_grad'])
                state_steps.append(state['step'])

    @torch.no_grad()
    def step(self, closure=None): # type: ignore
        self._cuda_graph_capture_health_check()
        assert closure is None, "Closure is not supported"

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            accum_grads = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_avg_sqs, accum_grads, state_steps
            )

            if len(state_steps) == 0:
                continue

            accum_adam_foreach(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                accum_grads,
                state_steps,
                beta1,
                beta2,
                group['lr'],
                group['weight_decay'],
                group['eps'],
                group['accum_iter']
            )

__all__ = [
    'optim_fn_adam',
    'optim_fn_adamw',
    'optim_fn_accum_adam',
    'optim_fn_accum_adamw',
    'lr_scheduler_fn_cosine_with_warmup',
    'AccumAdam',
    'AccumAdamW'
]
