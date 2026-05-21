from typing import Dict, Tuple

import torch


STEP_TYPE_VOCAB = {
    "condition_extraction": 0,
    "relation_setup": 1,
    "computation": 2,
    "verification": 3,
    "correction": 4,
    "general_reasoning": 5,
}


def get_step_type_vocab() -> Dict[str, int]:
    return dict(STEP_TYPE_VOCAB)


class StepDecoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_step_types: int,
        proj_dim: int = 512,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_step_types = int(num_step_types)
        self.proj_dim = int(proj_dim)

        self.norm = torch.nn.LayerNorm(self.hidden_size)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.type_head = torch.nn.Linear(self.hidden_size, self.num_step_types)
        self.proj_head = torch.nn.Linear(self.hidden_size, self.proj_dim)

        if self.proj_dim != self.hidden_size:
            projection = torch.empty(self.hidden_size, self.proj_dim)
            torch.nn.init.normal_(projection, mean=0.0, std=self.hidden_size ** -0.5)
            projection = torch.nn.functional.normalize(projection, dim=0)
            self.register_buffer("target_projection", projection)
        else:
            self.register_buffer("target_projection", torch.empty(0))

    def forward(self, z):
        h = self.dropout(self.norm(z))
        return {
            "type_logits": self.type_head(h),
            "step_proj": self.proj_head(h),
        }

    def project_targets(self, target_embeds: torch.Tensor) -> torch.Tensor:
        if self.proj_dim == self.hidden_size or self.target_projection.numel() == 0:
            return target_embeds
        return target_embeds @ self.target_projection.to(device=target_embeds.device, dtype=target_embeds.dtype)


def save_step_decoder(path: str, model: StepDecoder, metadata: Dict) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": {
            **(metadata or {}),
            "hidden_size": model.hidden_size,
            "num_step_types": model.num_step_types,
            "proj_dim": model.proj_dim,
            "step_type_vocab": get_step_type_vocab(),
        },
    }
    torch.save(payload, path)


def load_step_decoder(path: str, device) -> Tuple[StepDecoder, Dict]:
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    hidden_size = int(metadata.get("hidden_size"))
    num_step_types = int(metadata.get("num_step_types", len(get_step_type_vocab())))
    proj_dim = int(metadata.get("proj_dim", 512))
    decoder = StepDecoder(hidden_size=hidden_size, num_step_types=num_step_types, proj_dim=proj_dim)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    decoder.load_state_dict(state_dict, strict=False)
    decoder.to(device)
    return decoder, metadata
