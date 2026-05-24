import torch
from torch.special import logsumexp

def cross_entropy(
    predicted_logits: torch.Tensor, # (batch_size, vocab_size)
    targets: torch.Tensor, # (batch_size, )
) -> torch.Tensor:
    '''
    -log(exp(𝑜𝑖[𝑥𝑖+1])/∑(exp(𝑜𝑖[𝑎]))) = -(log(exp(𝑜𝑖[𝑥𝑖+1])) - log(∑(exp(𝑜𝑖[𝑎])))) = -𝑜𝑖[𝑥𝑖+1] + logsum(𝑜𝑖)
    '''
    batch_size = targets.size(0)
    log_Z = logsumexp(predicted_logits, dim=-1) # (batch_size, )
    log_p = predicted_logits[torch.arange(batch_size), targets] # (batch_size, )
    return (log_Z - log_p).mean() # scalar