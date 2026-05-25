import torch
from typing import Optional
from collections.abc import Callable
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {'lr': lr}
        super(SGD, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                lr = group['lr']
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1
        return loss

torch.random.manual_seed(42)
for lr in [1e3, 1e2, 1e1]:
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    print("====================================")
    for t in range(10):
        opt.zero_grad() # Reset param.grad
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

'''
====================================
24.16925811767578
8725.1025390625
1506962.375
167633472.0
13578309632.0
856946900992.0
43992855085056.0
1892760814616576.0
6.976312801912422e+16
2.2401715161488425e+18
====================================
19.545459747314453
19.545459747314453
3.35347056388855
0.08025608956813812
9.010370548188254e-17
1.0042617837271547e-18
3.381702398459162e-20
2.0145036680771744e-21
1.728171090366368e-22
1.9201903107699836e-23
====================================
17.2154483795166
11.017888069152832
8.121916770935059
6.354532241821289
5.1471710205078125
4.267594814300537
3.599149703979492
3.075576066970825
2.6560018062591553
2.3136727809906006
 *SGD lr tuning on toy problem*
       lr      behavior                      reason
      ----  -----------------  ------------------------------------------
      1e3   explodes           Step too large, gradient amplifies.
      1e2   converges fast     lr/sqrt(t+1) decay just right, w -> 0.
      1e1   descends steadily   Small safe steps, w -> 0 slowly.
'''