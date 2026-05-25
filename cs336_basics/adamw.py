import torch
from typing import Optional
from collections.abc import Callable
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

            eps = group['eps']
            weight_decay = group['weight_decay']
            state = self.state[p]
            alpha = group['lr']
            beta1, beta2 = group['betas']
            m = state.get('m_t', torch.zeros_like(p.data))
            v = state.get('v_t', torch.zeros_like(p.data))
            t = state.get('t', 1)
            beta1_pow = state.get('beta1_t', beta1)
            beta2_pow = state.get('beta2_t', beta2)
            g = p.grad.data
            alpha_t = alpha * math.sqrt(1 - beta2_pow) / (1 - beta1_pow)
            p.data -= alpha * weight_decay * p.data
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            p.data -= alpha_t * m / (torch.sqrt(v) + eps)
            state['m_t'] = m
            state['v_t'] = v
            state['beta1_t'] = beta1_pow * beta1
            state['beta2_t'] = beta2_pow * beta2
            state['t'] = t + 1
        return loss