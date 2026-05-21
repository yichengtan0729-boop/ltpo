import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ltpo import build_inputs
from scenario_router import DEFAULT_ROLES
from step_decoder import StepDecoder, get_step_type_vocab, save_step_decoder
from utils_io import ensure_dir, read_jsonl, safe_name_from_path, save_json


def _step_memory_path(args) -> str:
    if getattr(args, "step_memory_output_path", None):
        return args.step_memory_output_path
    data_name = safe_name_from_path(getattr(args, "memory_dataset", None) or args.dataset)
    model_name = safe_name_from_path(args.model_name_or_path)
    memory_dir = args.step_memory_dir or os.path.join(args.output_dir, "step_memories")
    return os.path.join(memory_dir, f"{model_name}-{data_name}-step-memory.jsonl")


def _decoder_path(args) -> str:
    if getattr(args, "step_decoder_path", None):
        return args.step_decoder_path
    data_name = safe_name_from_path(getattr(args, "memory_dataset", None) or args.dataset)
    model_name = safe_name_from_path(args.model_name_or_path)
    decoder_dir = args.step_decoder_dir or os.path.join(args.output_dir, "step_decoders")
    return os.path.join(decoder_dir, f"{model_name}-{data_name}-step-decoder.pt")


def _role_index(role: str, roles: List[str], num_tokens: int) -> int:
    if role in roles:
        return min(roles.index(role), max(0, num_tokens - 1))
    if role in DEFAULT_ROLES:
        return min(DEFAULT_ROLES.index(role), max(0, num_tokens - 1))
    return 0


def _mean_input_embedding(model, tokenizer, text: str, device, dtype=torch.float32) -> Optional[torch.Tensor]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized["input_ids"].to(device)
        if input_ids.numel() == 0:
            return None
        embeds = model.get_input_embeddings()(input_ids)
        return embeds.mean(dim=1).squeeze(0).detach().to(dtype=dtype)
    except Exception:
        return None


def _extract_latent_state(args, model, tokenizer, item: Dict[str, Any], target_embed: torch.Tensor, device) -> Optional[torch.Tensor]:
    try:
        num_tokens = max(1, int(getattr(args, "num_thought_tokens", 4)))
        inputs, thought_idx = build_inputs(
            tokenizer=tokenizer,
            num_thought_tokens=num_tokens,
            prompt=item.get("question", ""),
            device=device,
            data_name=getattr(args, "dataset", ""),
            model_name=getattr(args, "model_name_or_path", ""),
        )
        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
        roles = list(getattr(args, "step_roles", None) or DEFAULT_ROLES)
        pos = thought_idx[0] + _role_index(item.get("role", ""), roles, num_tokens)
        inputs_embeds[0, pos] = target_embed.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs["inputs_embeds"] = inputs_embeds
        inputs.pop("input_ids")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            z = outputs.hidden_states[-1][0, pos].detach().to(dtype=torch.float32)
        return z
    except Exception:
        return None


def _contrastive_loss(step_proj: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:
    if step_proj.shape[0] <= 1:
        return step_proj.new_tensor(0.0)
    logits = F.normalize(step_proj, dim=-1) @ F.normalize(target_proj, dim=-1).T
    logits = logits / 0.07
    labels = torch.arange(step_proj.shape[0], device=step_proj.device)
    return F.cross_entropy(logits, labels)


def _save_untrained_decoder(path: str, hidden_size: int, proj_dim: int, reason: str) -> str:
    ensure_dir(os.path.dirname(path))
    decoder = StepDecoder(hidden_size=hidden_size, num_step_types=len(get_step_type_vocab()), proj_dim=proj_dim)
    save_step_decoder(
        path,
        decoder,
        {
            "trained": False,
            "reason": reason,
            "loss_history": [],
        },
    )
    return path


def train_step_decoder(args, model, tokenizer) -> str:
    step_memory_path = _step_memory_path(args)
    output_path = _decoder_path(args)
    ensure_dir(os.path.dirname(output_path))

    device = next(model.parameters()).device
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    hidden_size = int(getattr(model.get_input_embeddings(), "embedding_dim", getattr(model.config, "hidden_size", 0)))
    proj_dim = int(getattr(args, "step_decoder_proj_dim", 512))
    vocab = get_step_type_vocab()

    if not os.path.exists(step_memory_path):
        return _save_untrained_decoder(output_path, hidden_size, proj_dim, f"missing step memory: {step_memory_path}")

    raw_items = read_jsonl(step_memory_path)
    items = [
        it for it in raw_items
        if it.get("memory_type") in {"verified_solution", "correction", "failure"}
        and str(it.get("step_text", "")).strip()
    ]
    if not items:
        return _save_untrained_decoder(output_path, hidden_size, proj_dim, "no usable step memory entries")

    decoder = StepDecoder(hidden_size=hidden_size, num_step_types=len(vocab), proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=float(getattr(args, "step_decoder_lr", 1e-3)))

    epochs = max(1, int(getattr(args, "step_decoder_epochs", 1)))
    batch_size = max(1, int(getattr(args, "step_decoder_batch_size", 1)))
    type_weight = float(getattr(args, "decoder_type_weight", 1.0))
    sem_weight = float(getattr(args, "decoder_sem_weight", 0.5))
    ctr_weight = float(getattr(args, "decoder_ctr_weight", 0.0))

    loss_history: List[Dict[str, float]] = []
    usable_count = 0

    for epoch in range(epochs):
        decoder.train()
        epoch_losses: List[float] = []

        for start in tqdm(range(0, len(items), batch_size), desc=f"Training step decoder epoch {epoch + 1}"):
            batch = items[start:start + batch_size]
            zs: List[torch.Tensor] = []
            targets: List[torch.Tensor] = []
            type_ids: List[int] = []

            for item in batch:
                target_embed = _mean_input_embedding(model, tokenizer, item.get("step_text", ""), device=device)
                if target_embed is None:
                    continue
                z = _extract_latent_state(args, model, tokenizer, item, target_embed, device=device)
                if z is None:
                    continue
                zs.append(z)
                targets.append(target_embed)
                type_ids.append(vocab.get(item.get("step_type", "general_reasoning"), vocab["general_reasoning"]))

            if not zs:
                continue

            usable_count += len(zs)
            z_tensor = torch.stack(zs, dim=0).to(device=device, dtype=torch.float32)
            target_tensor = torch.stack(targets, dim=0).to(device=device, dtype=torch.float32)
            type_tensor = torch.tensor(type_ids, device=device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = decoder(z_tensor)
            target_proj = decoder.project_targets(target_tensor)

            l_type = F.cross_entropy(outputs["type_logits"], type_tensor)
            l_sem = 1.0 - F.cosine_similarity(outputs["step_proj"], target_proj, dim=-1).mean()
            l_ctr = _contrastive_loss(outputs["step_proj"], target_proj) if ctr_weight > 0 else z_tensor.new_tensor(0.0)

            loss = type_weight * l_type + sem_weight * l_sem + ctr_weight * l_ctr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            loss_record = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu().item()),
                "type_loss": float(l_type.detach().cpu().item()),
                "semantic_loss": float(l_sem.detach().cpu().item()),
                "contrastive_loss": float(l_ctr.detach().cpu().item()),
            }
            loss_history.append(loss_record)
            epoch_losses.append(loss_record["loss"])

        if getattr(args, "verbose", 0):
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            print(f"[train_step_decoder] epoch={epoch + 1} mean_loss={mean_loss:.6f}")

    metadata = {
        "trained": usable_count > 0,
        "num_entries": len(items),
        "usable_entries": usable_count,
        "step_memory_path": step_memory_path,
        "loss_history": loss_history,
    }
    if usable_count == 0:
        metadata["reason"] = "no entries produced latent states"

    decoder.eval()
    save_step_decoder(output_path, decoder, metadata)
    save_json(os.path.splitext(output_path)[0] + ".loss.json", {"loss_history": loss_history, **metadata})
    return output_path
