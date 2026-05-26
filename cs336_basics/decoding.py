import torch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.softmax import softmax
import random

def decode(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens_num: int,
    temperature: float,
    top_p: float,
) -> str:
    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)  # shape (1, seq)
    if len(token_ids[0]) > model.context_length:
        token_ids = token_ids[:, -model.context_length:]
    eot_id = tokenizer.encode('<|endoftext|>')[0]
    i = 0
    output = []
    while i < max_tokens_num:
        logits = model(token_ids) # shape (1, seq, vocab_size)
        logits = logits[0, -1, :] # shape (vocab_size,)
        probs = softmax(logits / temperature, dim=-1) # shape (vocab_size,)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True) # shape (vocab_size,)

        cumsum = torch.cumsum(sorted_probs, dim=0) # shape (vocab_size,)
        cutoff = cumsum < top_p # shape (vocab_size,)
        keep = sorted_indices[cutoff] # set V(p)
        n_keep = len(keep)
        token_list = sorted_indices[:n_keep + 1].tolist()
        prob_tensor = sorted_probs[:n_keep + 1]
        prob_list = (prob_tensor / sum(prob_tensor)).tolist() # 概率归一化
        next_token = random.choices(token_list, weights=prob_list, k=1)[0]
        output.append(next_token)
        token_ids = torch.cat([token_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1) # 这里怎么办，应该加进去
        if len(token_ids[0]) > model.context_length:
            token_ids = token_ids[:, -model.context_length:]
        if next_token == eot_id:
            break
        i += 1
    return tokenizer.decode(output)