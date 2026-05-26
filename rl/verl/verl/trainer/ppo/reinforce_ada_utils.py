"""
Utility functions for DataProto manipulation in multi-round generation with adaptive downsampling.
"""

import numpy as np
import torch

import ray
from verl import DataProto
from verl.trainer.ppo.reward import compute_reward, compute_reward_async


def get_first_dim_size(dp: DataProto) -> int:
    """Get the batch size (first dimension) of a DataProto object.

    Args:
        dp: DataProto object to inspect

    Returns:
        The size of the first dimension (batch size)
    """
    if hasattr(dp, "batch") and isinstance(dp.batch, dict) and dp.batch:
        for v in dp.batch.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
    if hasattr(dp, "non_tensor_batch") and isinstance(dp.non_tensor_batch, dict) and dp.non_tensor_batch:
        for v in dp.non_tensor_batch.values():
            try:
                return len(v)
            except Exception:
                continue
    raise RuntimeError("Cannot infer batch size from DataProto")


def concat_dataproto_fragments(frags: list[DataProto]) -> DataProto:
    """Concatenate multiple DataProto fragments using efficient TensorDict structure.

    Args:
        frags: List of DataProto objects to concatenate

    Returns:
        Merged DataProto object
    """
    assert len(frags) > 0, "Empty fragment list"

    # Check key consistency (keep original warning mechanism)
    tensor_keys_sets = [set(f.batch.keys()) for f in frags]
    nontensor_keys_sets = [set(f.non_tensor_batch.keys()) for f in frags]
    tensor_keys = set.intersection(*tensor_keys_sets) if tensor_keys_sets else set()
    nontensor_keys = set.intersection(*nontensor_keys_sets) if nontensor_keys_sets else set()

    if any(set(f.batch.keys()) != tensor_keys for f in frags):
        missing = set.union(*tensor_keys_sets) - tensor_keys
        print(f"[warn] tensor keys inconsistent, using intersection: ignoring {missing}")
    if any(set(f.non_tensor_batch.keys()) != nontensor_keys for f in frags):
        missing = set.union(*nontensor_keys_sets) - nontensor_keys
        print(f"[warn] non-tensor keys inconsistent, using intersection: ignoring {missing}")

    # Use DataProto.concat() to maintain TensorDict optimization
    try:
        merged = DataProto.concat(frags)
        return merged
    except Exception as e:
        # Fallback to manual concatenation if concat fails
        print(f"[warn] DataProto.concat() failed, falling back to manual concat: {e}")
        out_batch = {k: torch.cat([f.batch[k] for f in frags], dim=0) for k in tensor_keys}
        out_non_tensor = {}
        for k in nontensor_keys:
            parts = [np.array(f.non_tensor_batch[k], dtype=object) for f in frags]
            out_non_tensor[k] = np.concatenate(parts, axis=0)
        merged = DataProto.from_single_dict({**out_batch, **out_non_tensor})
        try:
            merged.meta_info = dict(getattr(frags[0], "meta_info", {}) or {})
        except Exception:
            pass
        return merged


def build_uid_to_fields_mapping(context_batch: DataProto) -> dict:
    """Build a mapping from uid to all non-tensor fields for context alignment.

    Args:
        context_batch: DataProto containing context information with uid field

    Returns:
        Dictionary mapping uid -> {field_name: field_value}
    """
    if "uid" not in context_batch.non_tensor_batch:
        raise KeyError("context_batch missing uid; cannot align fields by uid")

    ctx_uid_to_fields = {}
    ctx_uids = list(context_batch.non_tensor_batch["uid"])
    ctx_keys = list(context_batch.non_tensor_batch.keys())

    for i, u in enumerate(ctx_uids):
        d = ctx_uid_to_fields.setdefault(u, {})
        for key in ctx_keys:
            d[key] = context_batch.non_tensor_batch[key][i]

    return ctx_uid_to_fields


def ensure_uid_in_batch(batch: DataProto, context_batch: DataProto = None) -> None:
    """Ensure that the batch has uid field, copying from context if needed.

    Args:
        batch: DataProto to ensure has uid
        context_batch: Optional context DataProto to copy uid from
    """
    if "uid" in batch.non_tensor_batch:
        return

    if context_batch is not None and "uid" in context_batch.non_tensor_batch:
        if get_first_dim_size(context_batch) == get_first_dim_size(batch):
            batch.non_tensor_batch["uid"] = np.array(list(context_batch.non_tensor_batch["uid"]), dtype=object)
            return

    raise KeyError("batch missing uid and cannot copy from context_batch; ensure _get_gen_batch passes through uid")


def align_context_to_selected(selected: DataProto, ctx: DataProto) -> DataProto:
    """Align context rows to match selected batch based on uid.

    Creates a view of context rows that matches the uid order in selected batch.
    Used for field completion without doing union.

    Args:
        selected: DataProto with uid field defining the desired order
        ctx: Context DataProto to align

    Returns:
        Context DataProto with rows reordered to match selected
    """
    if "uid" not in selected.non_tensor_batch:
        raise KeyError("selected_batch missing uid, cannot align context")
    sel_uids = list(selected.non_tensor_batch["uid"])

    if "uid" not in ctx.non_tensor_batch:
        raise KeyError("context_batch missing uid, cannot align")
    ctx_uids = list(ctx.non_tensor_batch["uid"])

    # Build uid -> first occurrence row index mapping
    uid_to_idx = {}
    for i, u in enumerate(ctx_uids):
        if u not in uid_to_idx:
            uid_to_idx[u] = i

    # Align to selected rows in order
    idxs = []
    miss = []
    for i, u in enumerate(sel_uids):
        j = uid_to_idx.get(u, None)
        if j is None:
            miss.append((i, u))
        else:
            idxs.append(j)

    if miss:
        examples = miss[:5]
        raise KeyError(f"Some uids not found in context_batch, examples: {examples}")

    return ctx[idxs]


def merge_context_fields_into_batch(batch: DataProto, ctx_rows: DataProto) -> None:
    """Merge missing fields from context into batch (in-place).

    Only adds fields that don't already exist in batch (no overwriting).

    Args:
        batch: Target DataProto to add fields to
        ctx_rows: Source DataProto with context fields (must be same size as batch)
    """
    # 1) Non-tensor keys
    for k, v in ctx_rows.non_tensor_batch.items():
        if k not in batch.non_tensor_batch:
            batch.non_tensor_batch[k] = v

    # 2) Tensor keys (only when batch doesn't have that tensor)
    ctx_batch = getattr(ctx_rows, "batch", None)
    if ctx_batch is not None:
        n_selected = get_first_dim_size(batch)
        # Compatible with dict / TensorDict: both have .items()
        for k, v in ctx_batch.items():
            if k in batch.batch:
                continue
            if v.shape[0] != n_selected:
                raise ValueError(f"ctx_rows.batch['{k}'] row count({v.shape[0]}) != batch({n_selected})")
            batch.batch[k] = v


def validate_tensordict_performance(batch: DataProto, context: str = "batch") -> None:
    """Check and warn if batch is not using efficient TensorDict structure.

    Args:
        batch: DataProto to validate
        context: Context string for logging (e.g., "batch", "final_batch")
    """
    if hasattr(batch.batch, "__class__"):
        batch_type = batch.batch.__class__.__name__
        if "TensorDict" not in batch_type and "dict" in batch_type.lower():
            print(f"[perf_warn] {context}.batch is plain {batch_type}, may impact performance")
        else:
            print(f"[perf_info] {context}.batch is efficient {batch_type}")


def compute_seq_rewards_for_round(
    mini_prompt_batch: DataProto,
    gen_out: DataProto,
    ctx_uid_to_fields: dict,
    reward_fn,
    use_rm: bool,
    rm_wg,
    config,
    kl_ctrl_in_reward=None,
) -> tuple[DataProto, torch.Tensor, list]:
    """Compute sequence-level rewards for a round of generation.

    Aligns fields between prompt batch and generation output, then computes rewards.

    Args:
        mini_prompt_batch: Prompt batch for this round
        gen_out: Generated output batch
        ctx_uid_to_fields: Mapping from uid to context fields
        reward_fn: Reward function to compute rewards
        use_rm: Whether to use reward model
        rm_wg: Reward model worker group
        config: Configuration object
        kl_ctrl_in_reward: Optional KL controller for reward penalty

    Returns:
        Tuple of (mini_with_rewards, seq_reward, uids_round) where:
            - mini_with_rewards: DataProto with computed rewards
            - seq_reward: Tensor of sequence-level rewards
            - uids_round: List of uids in this round
    """
    from verl.trainer.ppo.ray_trainer import apply_kl_penalty
    
    Bp = get_first_dim_size(mini_prompt_batch)
    Bg = get_first_dim_size(gen_out)
    if Bg % Bp != 0:
        raise ValueError(f"Batch mismatch: gen_out({Bg}) is not a multiple of mini_prompt_batch({Bp}).")
    rep = Bg // Bp

    if not hasattr(gen_out, "non_tensor_batch") or gen_out.non_tensor_batch is None:
        gen_out.non_tensor_batch = {}

    # 1) Align uid
    if "uid" not in gen_out.non_tensor_batch:
        if "uid" in mini_prompt_batch.non_tensor_batch:
            gen_out.non_tensor_batch["uid"] = np.repeat(
                np.array(mini_prompt_batch.non_tensor_batch["uid"], dtype=object), rep, axis=0
            )
        else:
            raise KeyError("Cannot align uid in gen_out; mini_prompt_batch also missing uid")

    # 2) Copy all non-tensor keys from mini_prompt_batch (expanded by rep)
    for k, v in mini_prompt_batch.non_tensor_batch.items():
        if k in gen_out.non_tensor_batch:
            continue
        arr = np.array(v, dtype=object)
        if arr.shape[0] != Bp:
            raise ValueError(f"mini_prompt_batch.non_tensor_batch['{k}'] length {arr.shape[0]} != {Bp}")
        gen_out.non_tensor_batch[k] = np.repeat(arr, rep, axis=0)

    # 3) Fill required fields from context (uid-based join)
    uids_round = list(gen_out.non_tensor_batch["uid"])
    required_keys = ["reward_model"]
    rfk = getattr(reward_fn, "reward_fn_key", None)
    if isinstance(rfk, str) and len(rfk) > 0:
        required_keys.append(rfk)
    else:
        required_keys.append("data_source")

    for key in required_keys:
        if key in gen_out.non_tensor_batch:
            continue
        filled, miss = [], 0
        for u in uids_round:
            src = ctx_uid_to_fields.get(u, None)
            if src is None or key not in src:
                miss += 1
                filled.append(None)
            else:
                filled.append(src[key])
        if miss == len(uids_round):
            raise KeyError(f"Required field '{key}' not found in mini_prompt_batch or context_batch")
        if any(x is None for x in filled):
            ids = [i for i, x in enumerate(filled) if x is None][:5]
            raise KeyError(
                f"'{key}' still has missing values via uid mapping (example indices: {ids}). "
                "Ensure context_batch covers all active uids."
            )
        gen_out.non_tensor_batch[key] = np.array(filled, dtype=object)

    # 4) Fill auxiliary fields from context if available
    if ctx_uid_to_fields:
        sample_any = next(iter(ctx_uid_to_fields.values()), {})
        ctx_all_keys = set(sample_any.keys()) if isinstance(sample_any, dict) else set()
        aux_keys = [k for k in ctx_all_keys if k not in gen_out.non_tensor_batch]
        for key in aux_keys:
            try:
                filled = [ctx_uid_to_fields.get(u, {}).get(key, None) for u in uids_round]
                if all(v is None for v in filled):
                    continue
                gen_out.non_tensor_batch[key] = np.array(filled, dtype=object)
            except Exception:
                pass

    # 5) Copy meta_info if needed
    if hasattr(mini_prompt_batch, "meta_info") and isinstance(mini_prompt_batch.meta_info, dict):
        if not hasattr(gen_out, "meta_info") or gen_out.meta_info is None:
            gen_out.meta_info = {}
        if "global_steps" in mini_prompt_batch.meta_info and "global_steps" not in gen_out.meta_info:
            gen_out.meta_info["global_steps"] = mini_prompt_batch.meta_info["global_steps"]

    # 6) Compute rewards
    mini = gen_out
    if use_rm and "rm_scores" not in mini.batch.keys():
        rm_tensor = rm_wg.compute_rm_score(mini)
        mini = mini.union(rm_tensor)

    if config.reward_model.launch_reward_fn_async:
        reward_tensor, reward_extra_infos_dict = ray.get(compute_reward_async.remote(data=mini, reward_fn=reward_fn))
    else:
        reward_tensor, reward_extra_infos_dict = compute_reward(mini, reward_fn)

    mini.batch["token_level_scores"] = reward_tensor

    if config.algorithm.use_kl_in_reward:
        mini, _ = apply_kl_penalty(mini, kl_ctrl=kl_ctrl_in_reward, kl_penalty=config.algorithm.kl_penalty)
        seq_reward = mini.batch["token_level_rewards"].sum(dim=-1)
    else:
        seq_reward = reward_tensor.sum(dim=-1)
        mini.batch["token_level_rewards"] = reward_tensor

    if reward_extra_infos_dict:
        for k, v in reward_extra_infos_dict.items():
            try:
                if len(v) == get_first_dim_size(mini):
                    mini.non_tensor_batch[k] = np.array(v, dtype=object)
            except Exception:
                pass

    return mini, seq_reward, uids_round
