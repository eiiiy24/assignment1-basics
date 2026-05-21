import torch
from einops import rearrange, einsum

class Embedding(torch.nn.Module):
    """An embedding lookup layer that maps integer token IDs to dense vectors.

    Stores a weight matrix of shape (num_embeddings, embedding_dim) where each row
    is the embedding vector for the corresponding token.
    """

    def __init__(
        self,
        num_embeddings: int, # Size of the vocabulary
        embedding_dim: int, # Dimension of the embedding vectors, i.e., 𝑑model
        device: torch.device | None=None, # Device to store the parameters on
        dtype: torch.dtype | None=None # Data type of the parameters
    ):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3) # Embedding: 𝒩︀(𝜇 = 0,𝜎2 = 1) truncated at [−3,3]

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs.
        return self.weight[token_ids] # (batch_size, sequence_length) -> (batch_size, sequence_length, d_model)