import torch
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerLM(torch.nn.Module):
    r"""Transformer language model.

    Token embeddings → ``num_layers`` pre-norm Transformer blocks → final RMSNorm
    → LM head linear projection, producing unnormalized logits over the vocabulary.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        vocab_size: int, # The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix
        context_length: int, # The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer
        num_layers: int, # The number of Transformer blocks to use
        use_norm: bool = True,  # Set False for RMSNorm ablation
        norm_position: str = "pre",  # "pre" or "post"
        use_rope: bool = True,  # Set False for NoPE ablation
        use_gate: bool = True,  # Set False for GLU ablation
        device: torch.device | None=None,
        dtype: torch.dtype | None=None,
    ):
        super(TransformerLM, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.use_norm = use_norm
        self.norm_position = norm_position
        self.use_rope = use_rope
        self.use_gate = use_gate
        self.device = device
        self.dtype = dtype
        self.emd = Embedding(self.vocab_size, self.d_model, device=self.device, dtype=self.dtype)
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                self.d_model,
                self.num_heads, 
                self.d_ff,
                self.rope_theta,
                self.context_length,
                use_norm=self.use_norm,
                norm_position=self.norm_position,
                use_rope=self.use_rope,
                use_gate=self.use_gate,
                device=self.device,
                dtype=self.dtype)
            for _ in range(self.num_layers)
        ])
        if self.use_norm:
            self.norm = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.linear = Linear(self.d_model, self.vocab_size, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emd(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x) if self.use_norm else x
        x = self.linear(x)
        return x
