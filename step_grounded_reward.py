from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ltpo import get_confidence


class StepGroundedReward:
    def __init__(
        self,
        model,
        tokenizer,
        thought_idx,
        inputs,
        target_step_embeds,
        step_decoder=None,
        step_type_ids=None,
        failure_embeds=None,
        top_k=10,
        conf_weight=0.4,
        align_weight=0.4,
        collapse_weight=0.2,
        decoder_weight=0.2,
        failure_weight=0.2,
        collapse_margin=0.5,
        disable_step_decoder=False,
        disable_failure_penalty=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.thought_idx = thought_idx
        self.inputs = inputs
        self.target_step_embeds = target_step_embeds.detach() if target_step_embeds is not None else None
        self.step_decoder = step_decoder
        self.step_type_ids = step_type_ids.detach().long() if step_type_ids is not None else None
        self.failure_embeds = failure_embeds.detach() if failure_embeds is not None else None
        self.top_k = int(top_k or 10)
        self.conf_weight = float(conf_weight)
        self.align_weight = float(align_weight)
        self.collapse_weight = float(collapse_weight)
        self.decoder_weight = float(decoder_weight)
        self.failure_weight = float(failure_weight)
        self.collapse_margin = float(collapse_margin)
        self.disable_step_decoder = bool(disable_step_decoder)
        self.disable_failure_penalty = bool(disable_failure_penalty)

        if self.step_decoder is not None:
            self.step_decoder.eval()
            for param in self.step_decoder.parameters():
                param.requires_grad_(False)

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(0.0)

    def _step_alignment(self, thought_hidden_states: torch.Tensor) -> torch.Tensor:
        if self.target_step_embeds is None or self.target_step_embeds.numel() == 0:
            return self._zero(thought_hidden_states)
        k = min(thought_hidden_states.shape[0], self.target_step_embeds.shape[0])
        if k <= 0:
            return self._zero(thought_hidden_states)
        z = thought_hidden_states[:k]
        target = self.target_step_embeds[:k].to(device=z.device, dtype=z.dtype)
        return F.cosine_similarity(z, target, dim=-1).mean()

    def _collapse_penalty(self, thought_hidden_states: torch.Tensor) -> torch.Tensor:
        if thought_hidden_states.shape[0] <= 1:
            return self._zero(thought_hidden_states)
        z = F.normalize(thought_hidden_states, dim=-1)
        sim = z @ z.T
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        off_diag = sim[mask]
        penalty = torch.clamp(off_diag - self.collapse_margin, min=0.0).pow(2).mean()
        return penalty

    def _decoder_validity(self, thought_hidden_states: torch.Tensor) -> torch.Tensor:
        if self.disable_step_decoder or self.step_decoder is None:
            return self._zero(thought_hidden_states)
        if self.step_type_ids is None or self.step_type_ids.numel() == 0:
            return self._zero(thought_hidden_states)
        if self.target_step_embeds is None or self.target_step_embeds.numel() == 0:
            return self._zero(thought_hidden_states)

        k = min(
            thought_hidden_states.shape[0],
            self.step_type_ids.shape[0],
            self.target_step_embeds.shape[0],
        )
        if k <= 0:
            return self._zero(thought_hidden_states)

        z = thought_hidden_states[:k]
        try:
            decoder_dtype = next(self.step_decoder.parameters()).dtype
        except StopIteration:
            decoder_dtype = z.dtype
        z_dec = z.to(dtype=decoder_dtype)

        outputs = self.step_decoder(z_dec)
        type_ids = self.step_type_ids[:k].to(device=z_dec.device)
        type_score = F.log_softmax(outputs["type_logits"], dim=-1).gather(1, type_ids[:, None]).mean()

        target = self.target_step_embeds[:k].to(device=z_dec.device, dtype=decoder_dtype)
        if hasattr(self.step_decoder, "project_targets"):
            target_proj = self.step_decoder.project_targets(target)
        else:
            target_proj = target
        if target_proj.shape[-1] != outputs["step_proj"].shape[-1]:
            dim = min(target_proj.shape[-1], outputs["step_proj"].shape[-1])
            target_proj = target_proj[..., :dim]
            step_proj = outputs["step_proj"][..., :dim]
        else:
            step_proj = outputs["step_proj"]
        semantic_score = F.cosine_similarity(step_proj, target_proj, dim=-1).mean()
        decoder_validity = torch.nan_to_num(type_score + semantic_score)
        return decoder_validity.to(dtype=thought_hidden_states.dtype)

    def _failure_penalty(self, thought_hidden_states: torch.Tensor) -> torch.Tensor:
        if self.disable_failure_penalty or self.failure_embeds is None or self.failure_embeds.numel() == 0:
            return self._zero(thought_hidden_states)
        trajectory = F.normalize(thought_hidden_states.mean(dim=0, keepdim=True), dim=-1)
        failures = self.failure_embeds.to(device=thought_hidden_states.device, dtype=thought_hidden_states.dtype)
        failures = F.normalize(failures, dim=-1)
        sims = trajectory @ failures.T
        return sims.max()

    def compute(self, thought_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        local_inputs = dict(self.inputs)
        if "inputs_embeds" in local_inputs:
            local_inputs["inputs_embeds"] = local_inputs["inputs_embeds"].detach().clone()
        confidence = get_confidence(
            model=self.model,
            inputs=local_inputs,
            thought_idx=self.thought_idx,
            thought_hidden_states=thought_hidden_states,
            k=self.top_k,
        )
        confidence = torch.nan_to_num(confidence)

        step_alignment = torch.nan_to_num(self._step_alignment(thought_hidden_states))
        collapse_penalty = torch.nan_to_num(self._collapse_penalty(thought_hidden_states))
        decoder_validity = torch.nan_to_num(self._decoder_validity(thought_hidden_states))
        failure_penalty = torch.nan_to_num(self._failure_penalty(thought_hidden_states))

        reward = (
            self.conf_weight * confidence
            + self.align_weight * step_alignment
            - self.collapse_weight * collapse_penalty
            + self.decoder_weight * decoder_validity
            - self.failure_weight * failure_penalty
        )
        reward = torch.nan_to_num(reward)

        breakdown = {
            "confidence": float(confidence.detach().cpu().item()),
            "step_alignment": float(step_alignment.detach().cpu().item()),
            "collapse_penalty": float(collapse_penalty.detach().cpu().item()),
            "decoder_validity": float(decoder_validity.detach().cpu().item()),
            "failure_penalty": float(failure_penalty.detach().cpu().item()),
            "total_reward": float(reward.detach().cpu().item()),
        }
        return reward, breakdown
