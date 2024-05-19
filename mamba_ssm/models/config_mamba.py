import json
from dataclasses import dataclass, field, asdict


@dataclass
class MambaConfig:

    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    def to_json_string(self):
        """Serializes the dataclass instance to a JSON string."""
        return json.dumps(asdict(self), indent=4)
