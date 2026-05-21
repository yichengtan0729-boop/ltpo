import os
from typing import Dict, List, Optional, Tuple

import torch

from ltpo import build_inputs
from scenario_router import DEFAULT_ROLES, infer_scenario
from step_decoder import get_step_type_vocab, load_step_decoder
from step_grounded_reward import StepGroundedReward
from step_retriever import StepPrototypeRetriever


def _get_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prototype_text(proto: Dict) -> str:
    pieces = [str(proto.get("prototype_text", "") or "")]
    common_steps = proto.get("common_steps", []) or []
    if not isinstance(common_steps, list):
        common_steps = [str(common_steps)]
    pieces.extend(str(x) for x in common_steps if str(x).strip())
    return " ".join(pieces).strip()


def _mean_input_embedding(model, tokenizer, text: str, device, dtype) -> torch.Tensor:
    text = (text or "").strip()
    hidden_size = model.get_input_embeddings().weight.shape[-1]
    if not text:
        return torch.zeros(hidden_size, device=device, dtype=dtype)
    try:
        tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized["input_ids"].to(device)
        if input_ids.numel() == 0:
            return torch.zeros(hidden_size, device=device, dtype=dtype)
        with torch.no_grad():
            embeds = model.get_input_embeddings()(input_ids).mean(dim=1).squeeze(0)
        return embeds.detach().to(device=device, dtype=dtype)
    except Exception:
        return torch.zeros(hidden_size, device=device, dtype=dtype)


def _prepare_roles(step_roles: Optional[List[str]], num_thought_tokens: int) -> List[str]:
    roles = list(step_roles or DEFAULT_ROLES)
    for role in DEFAULT_ROLES:
        if len(roles) >= num_thought_tokens:
            break
        if role not in roles:
            roles.append(role)
    if len(roles) < num_thought_tokens:
        roles.extend([DEFAULT_ROLES[-1]] * (num_thought_tokens - len(roles)))
    return roles[:num_thought_tokens]


def _load_decoder_if_available(path: str, device, disabled: bool, verbose: int):
    if disabled or not path or not os.path.exists(path):
        return None, {}
    try:
        decoder, metadata = load_step_decoder(path, device=device)
        if metadata.get("trained") is False:
            if verbose:
                print("[step_memory_ltpo] decoder checkpoint is marked trained=false; falling back to V1 reward.")
            return None, metadata
        decoder.eval()
        for param in decoder.parameters():
            param.requires_grad_(False)
        return decoder, metadata
    except Exception as exc:
        if verbose:
            print(f"[step_memory_ltpo] decoder disabled: {exc}")
        return None, {}


def _build_latent_cot(retrieved: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for proto in retrieved:
        out.append(
            {
                "role": proto.get("role", ""),
                "step_type": proto.get("step_type", ""),
                "prototype_id": proto.get("prototype_id", ""),
                "prototype_text": proto.get("prototype_text", ""),
                "common_steps": proto.get("common_steps", []),
                "retrieval_score": float(proto.get("retrieval_score", 0.0)),
                "reliability": float(proto.get("reliability", proto.get("reliability_mean", 0.0))),
            }
        )
    return out


def generate_step_grounded(
    tokenizer,
    model,
    question: str,
    data_name: str,
    model_name: str,
    step_prototype_path: str,
    step_vectorizer_path: str = "",
    step_decoder_path: str = "",
    num_thought_tokens: int = 4,
    max_rl_steps: int = 5,
    max_new_tokens: int = 1024,
    lr: float = 0.03,
    sigma: float = 0.05,
    sigma_decay: float = 0.99,
    top_k: int = 10,
    step_roles: Optional[List[str]] = None,
    step_top_k_per_role: int = 1,
    step_conf_weight: float = 0.4,
    step_align_weight: float = 0.4,
    step_collapse_weight: float = 0.2,
    step_decoder_weight: float = 0.2,
    step_failure_weight: float = 0.2,
    collapse_margin: float = 0.5,
    step_grounding_mix: float = 0.5,
    reward_threshold: float = -1,
    disable_step_decoder: bool = False,
    disable_failure_penalty: bool = False,
    verbose: int = 1,
    **kwargs,
) -> Tuple[str, float, int, List[Dict], List[Dict], Dict]:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    device = _get_device(model)
    scenario = infer_scenario(question, data_name)
    roles = _prepare_roles(step_roles, num_thought_tokens)

    retrieved_step_prototypes: List[Dict] = []
    failure_prototypes: List[Dict] = []
    try:
        retriever = StepPrototypeRetriever(step_prototype_path, step_vectorizer_path)
        retrieved_step_prototypes = retriever.retrieve_step_plan(
            question=question,
            scenario=scenario,
            roles=roles,
            top_k_per_role=step_top_k_per_role,
            memory_type="verified_solution",
        )
        if not disable_failure_penalty:
            failure_prototypes = retriever.retrieve_failure_prototypes(
                question=question,
                scenario=scenario,
                top_k=4,
            )
    except Exception as exc:
        if verbose:
            print(f"[step_memory_ltpo] step prototype retrieval disabled: {exc}")
        retrieved_step_prototypes = []
        failure_prototypes = []

    inputs, thought_idx = build_inputs(
        tokenizer=tokenizer,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        device=device,
        data_name=data_name,
        model_name=model_name,
    )
    inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
    inputs["inputs_embeds"] = inputs_embeds
    inputs.pop("input_ids")

    base_thought = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone().detach()
    embed_dtype = base_thought.dtype
    hidden_size = base_thought.shape[-1]

    first_proto_by_role: Dict[str, Dict] = {}
    for proto in retrieved_step_prototypes:
        role = proto.get("role", "")
        if role not in first_proto_by_role:
            first_proto_by_role[role] = proto

    target_embeds: List[torch.Tensor] = []
    step_type_ids: List[int] = []
    vocab = get_step_type_vocab()
    has_retrieved_target = bool(first_proto_by_role)

    for idx, role in enumerate(roles):
        proto = first_proto_by_role.get(role)
        if proto is None:
            target_embeds.append(base_thought[idx].detach())
            step_type_ids.append(vocab["general_reasoning"])
            continue
        text = _prototype_text(proto)
        target_embeds.append(_mean_input_embedding(model, tokenizer, text, device=device, dtype=embed_dtype))
        step_type_ids.append(vocab.get(proto.get("step_type", "general_reasoning"), vocab["general_reasoning"]))

    if target_embeds:
        target_step_embeds = torch.stack(target_embeds, dim=0).to(device=device, dtype=embed_dtype)
    else:
        target_step_embeds = torch.empty(0, hidden_size, device=device, dtype=embed_dtype)

    reward_target_embeds = target_step_embeds if has_retrieved_target else torch.empty(0, hidden_size, device=device, dtype=embed_dtype)
    step_type_tensor = torch.tensor(step_type_ids, device=device, dtype=torch.long) if step_type_ids else None

    failure_embeds_list = [
        _mean_input_embedding(model, tokenizer, _prototype_text(proto), device=device, dtype=embed_dtype)
        for proto in failure_prototypes
    ]
    if failure_embeds_list and not disable_failure_penalty:
        failure_embeds = torch.stack(failure_embeds_list, dim=0).to(device=device, dtype=embed_dtype)
    else:
        failure_embeds = torch.empty(0, hidden_size, device=device, dtype=embed_dtype)

    step_decoder, decoder_metadata = _load_decoder_if_available(
        step_decoder_path,
        device=device,
        disabled=disable_step_decoder,
        verbose=verbose,
    )
    decoder_enabled = step_decoder is not None

    if has_retrieved_target:
        mix = max(0.0, min(1.0, float(step_grounding_mix)))
        init = (1.0 - mix) * base_thought + mix * target_step_embeds.detach()
    else:
        init = base_thought

    thought_hidden_states = torch.nn.Parameter(init.clone().detach().requires_grad_(True))
    optimizer = torch.optim.Adam([thought_hidden_states], lr=float(lr))
    reward_model = StepGroundedReward(
        model=model,
        tokenizer=tokenizer,
        thought_idx=thought_idx,
        inputs=inputs,
        target_step_embeds=reward_target_embeds,
        step_decoder=step_decoder,
        step_type_ids=step_type_tensor,
        failure_embeds=failure_embeds,
        top_k=top_k,
        conf_weight=step_conf_weight,
        align_weight=step_align_weight,
        collapse_weight=step_collapse_weight,
        decoder_weight=step_decoder_weight,
        failure_weight=step_failure_weight,
        collapse_margin=collapse_margin,
        disable_step_decoder=not decoder_enabled,
        disable_failure_penalty=disable_failure_penalty,
    )

    best_reward = -float("inf")
    best_reward_step = -1
    best_breakdown: Dict[str, float] = {}
    best_thought_hidden_states = thought_hidden_states.detach().clone()
    reward_trace: List[Dict] = []
    cur_sigma = float(sigma)

    for step in range(max(0, int(max_rl_steps))):
        optimizer.zero_grad()
        epsilon = torch.normal(
            mean=0.0,
            std=max(cur_sigma, 0.0),
            size=thought_hidden_states.shape,
            device=device,
            dtype=thought_hidden_states.dtype,
        )
        candidate = thought_hidden_states + epsilon
        reward, breakdown = reward_model.compute(candidate)
        loss = -reward
        loss.backward()
        optimizer.step()

        reward_value = float(reward.detach().cpu().item())
        trace_item = {"step": step, **breakdown}
        reward_trace.append(trace_item)

        if verbose:
            print(f"[step_memory_ltpo] step={step} reward={reward_value:.6f} breakdown={breakdown}")

        if reward_value > best_reward:
            best_reward = reward_value
            best_reward_step = step
            best_breakdown = dict(breakdown)
            best_thought_hidden_states = candidate.detach().clone()

        cur_sigma *= float(sigma_decay)
        if reward_threshold > 0 and reward_value >= reward_threshold:
            break

    if best_reward_step < 0:
        best_thought_hidden_states = thought_hidden_states.detach().clone()
        best_reward = 0.0
        best_reward_step = 0
        best_breakdown = {
            "confidence": 0.0,
            "step_alignment": 0.0,
            "collapse_penalty": 0.0,
            "decoder_validity": 0.0,
            "failure_penalty": 0.0,
            "total_reward": 0.0,
        }

    inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states.to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    inputs["inputs_embeds"] = inputs_embeds

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=1,
    )
    gen_kwargs.update(kwargs)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    diagnostics = {
        "scenario": scenario,
        "step_alignment": float(best_breakdown.get("step_alignment", 0.0)),
        "collapse_penalty": float(best_breakdown.get("collapse_penalty", 0.0)),
        "decoder_validity": float(best_breakdown.get("decoder_validity", 0.0)),
        "failure_penalty": float(best_breakdown.get("failure_penalty", 0.0)),
        "num_retrieved_step_prototypes": len(retrieved_step_prototypes),
        "num_failure_prototypes": len(failure_prototypes),
        "step_roles": roles,
        "decoder_enabled": bool(decoder_enabled),
        "failure_penalty_enabled": bool(not disable_failure_penalty and len(failure_embeds_list) > 0),
        "decoder_metadata": decoder_metadata,
        "latent_cot": _build_latent_cot(retrieved_step_prototypes),
        "retrieved_step_prototypes": retrieved_step_prototypes,
        "failure_prototypes": failure_prototypes,
    }

    return (
        response,
        best_reward,
        best_reward_step,
        retrieved_step_prototypes,
        reward_trace,
        diagnostics,
    )
