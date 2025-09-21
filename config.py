from dataclasses import dataclass

@dataclass
class GPT2Config:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 784
    dropout: float = 0.1
    d_ff: int | None = None
    activation_function: str = 'gelu'
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if not isinstance(self.block_size, int) or self.block_size <= 0:
            raise ValueError("block_size must be an integer > 0")
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ValueError("vocab_size must be an integer > 0")
        if not isinstance(self.n_layer, int) or self.n_layer <= 0:
            raise ValueError("n_layer must be an integer > 0")
        if not isinstance(self.n_head, int) or self.n_head <= 0:
            raise ValueError("n_head must be an integer > 0")
        if not isinstance(self.d_model, int) or self.d_model <= 0:
            raise ValueError("d_model must be an integer > 0")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if not isinstance(self.d_ff, int) or self.d_ff <= 0:
            raise ValueError("d_ff must be an integer > 0")
        if not isinstance(self.dropout, float) or not (0 <= self.dropout <= 1):
            raise ValueError("dropout must be a float between 0 and 1")
        if self.activation_function not in ('gelu', 'relu', 'tanh', 'sigmoid'):
            raise ValueError("activation_function must be one of: 'gelu', 'relu', 'tanh', 'sigmoid'")