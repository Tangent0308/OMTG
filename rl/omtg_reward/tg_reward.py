"""
Temporal grounding reward functions for OMTG.

This is the open-source version of the original internal reward stack. The main
change is that caption reward no longer depends on ttlive_strategy_agent. It now
uses a small OpenAI-compatible judge client from `omtg_judge`.
"""

import json
import os
import re
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from omtg_judge.openai_client import OpenAIJudge

rm_name = os.getenv("RM_NAME", "Qwen3-30B-A3B")
debug_print = os.getenv("DEBUG_PRINT", "false").lower() == "true"

CAPTION_COVERAGE_WEIGHT = float(os.getenv("CAPTION_COVERAGE_WEIGHT", "0.35"))
CAPTION_PRECISION_WEIGHT = float(os.getenv("CAPTION_PRECISION_WEIGHT", "0.20"))
CAPTION_DISCRIMINABILITY_WEIGHT = float(os.getenv("CAPTION_DISCRIMINABILITY_WEIGHT", "0.15"))
CAPTION_COUNTERFACTUAL_WEIGHT = float(os.getenv("CAPTION_COUNTERFACTUAL_WEIGHT", "0.30"))

THINK_LENGTH_SOFT_THRESHOLD = int(os.getenv("THINK_LENGTH_SOFT_THRESHOLD", "2000"))
THINK_LENGTH_HARD_THRESHOLD = int(os.getenv("THINK_LENGTH_HARD_THRESHOLD", "5000"))
CAPTION_LENGTH_SOFT_THRESHOLD = int(os.getenv("CAPTION_LENGTH_SOFT_THRESHOLD", "100"))
CAPTION_LENGTH_HARD_THRESHOLD = int(os.getenv("CAPTION_LENGTH_HARD_THRESHOLD", "200"))

THINK_OVERLONG_PENALTY_FACTOR = float(os.getenv("THINK_OVERLONG_PENALTY_FACTOR", "1.0"))
CAPTION_OVERLONG_PENALTY_FACTOR = float(os.getenv("CAPTION_OVERLONG_PENALTY_FACTOR", "0.5"))
RECALL_TIOU_THRESHOLD = float(os.getenv("RECALL_TIOU_THRESHOLD", "0.5"))
PRF_THRESHOLDS = [0.3, 0.5, 0.7]
MAX_RETRIES = 3

_JUDGE_CLIENT: OpenAIJudge | None = None


def _get_judge_client() -> OpenAIJudge:
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        _JUDGE_CLIENT = OpenAIJudge.from_env(default_model=rm_name)
    return _JUDGE_CLIENT


@dataclass
class RewardWeights:
    tiou: float = 1.0
    format: float = 1.0
    caption: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    cacc: float = 0.0
    length_penalty: float = 0.0


class TGRewardStrategy(ABC):
    @abstractmethod
    def get_enabled_rewards(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_weights(self) -> RewardWeights:
        raise NotImplementedError

    def get_name(self) -> str:
        return self.__class__.__name__


class _BaseStrategy(TGRewardStrategy):
    def __init__(self, enabled: List[str], **weights: float):
        self.enabled = enabled
        self.weights = weights

    def get_enabled_rewards(self) -> List[str]:
        return self.enabled

    def get_weights(self) -> RewardWeights:
        return RewardWeights(**self.weights)


def _normalize(weights: Dict[str, float], fields: List[str]) -> Dict[str, float]:
    total = sum(weights[name] for name in fields)
    if total <= 0:
        return {name: 1.0 / len(fields) for name in fields}
    return {name: weights[name] / total for name in fields}


def get_current_strategy() -> TGRewardStrategy:
    strategy = os.getenv("TG_REWARD_STRATEGY", "tiou_format").lower().strip()
    config = {
        "tiou": float(os.getenv("TG_TIOU_WEIGHT", "1.0")),
        "format": float(os.getenv("TG_FORMAT_WEIGHT", "1.0")),
        "caption": float(os.getenv("TG_CAPTION_WEIGHT", "1.0")),
        "recall": float(os.getenv("TG_RECALL_WEIGHT", "1.0")),
        "precision": float(os.getenv("TG_PRECISION_WEIGHT", "1.0")),
        "f1": float(os.getenv("TG_F1_WEIGHT", "1.0")),
        "cacc": float(os.getenv("TG_CACC_WEIGHT", "1.0")),
        "length_penalty": float(os.getenv("TG_LENGTH_PENALTY_WEIGHT", "0.3")),
    }
    normalized = strategy.replace("_", "").replace("-", "").lower()

    if normalized == "tiouformat":
        weights = _normalize(config, ["tiou", "format"])
        return _BaseStrategy(["tiou", "format"], **weights)
    if normalized == "tiouformatcaption":
        weights = _normalize(config, ["tiou", "format", "caption"])
        return _BaseStrategy(["tiou", "format", "caption"], **weights)
    if normalized == "tiouformatcaptionlength":
        weights = _normalize(config, ["tiou", "format", "caption"])
        weights["length_penalty"] = config["length_penalty"]
        return _BaseStrategy(["tiou", "format", "caption", "length_penalty"], **weights)
    if normalized == "tiouformatrecall":
        weights = _normalize(config, ["tiou", "format", "recall"])
        return _BaseStrategy(["tiou", "format", "recall"], **weights)
    if normalized == "tiouformatrecallavg":
        weights = _normalize(config, ["tiou", "format", "recall"])
        return _BaseStrategy(["tiou", "format", "recall_avg"], **weights)
    if normalized == "tiouformatprecision":
        weights = _normalize(config, ["tiou", "format", "precision"])
        return _BaseStrategy(["tiou", "format", "precision"], **weights)
    if normalized == "tiouformatf1":
        weights = _normalize(config, ["tiou", "format", "f1"])
        return _BaseStrategy(["tiou", "format", "f1"], **weights)
    if normalized == "tiouformatprf":
        weights = _normalize(config, ["tiou", "format", "precision", "recall", "f1"])
        return _BaseStrategy(["tiou", "format", "precision", "recall_avg", "f1"], **weights)
    if normalized == "tiouformatcacc":
        weights = _normalize(config, ["tiou", "format", "cacc"])
        return _BaseStrategy(["tiou", "format", "cacc"], **weights)
    if normalized == "tiouformatf1cacc":
        weights = _normalize(config, ["tiou", "format", "f1", "cacc"])
        return _BaseStrategy(["tiou", "format", "f1", "cacc"], **weights)
    if normalized == "tiouformatf1cacccaptionlength":
        weights = _normalize(config, ["tiou", "format", "f1", "cacc", "caption"])
        weights["length_penalty"] = config["length_penalty"]
        return _BaseStrategy(["tiou", "format", "f1", "cacc", "caption", "length_penalty"], **weights)
    raise ValueError(f"Unknown reward strategy: {strategy}")


def extract_time_intervals(sentence: str, only_result: bool = False) -> List[List[float]]:
    if only_result:
        think_end_match = re.search(r"</think>", sentence, re.I)
        if think_end_match:
            sentence = sentence[think_end_match.end() :]

    intervals = []
    time_blocks = re.findall(r"<time>(.*?)</time>", sentence, flags=re.I)
    if time_blocks:
        for blk in time_blocks:
            m = re.fullmatch(
                r"\s*(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s)?\s*",
                blk.strip(),
                flags=re.I,
            )
            if m:
                intervals.append([float(m.group(1)), float(m.group(2))])
        if intervals:
            return intervals

    for match in re.findall(r"[Ff]rom\s+(\d+(?:\.\d+)?)\s*s?\s+to\s+(\d+(?:\.\d+)?)\s*s?", sentence):
        intervals.append([float(match[0]), float(match[1])])
    if intervals:
        return intervals

    for match in re.findall(r"(\d+(?:\.\d+)?)\s*s?\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*s?", sentence):
        intervals.append([float(match[0]), float(match[1])])
    return intervals


def extract_think_content(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.I)
    return match.group(1).strip() if match else ""


def extract_captions_with_timestamps(think_content: str) -> List[Dict[str, Any]]:
    captions = []
    pattern = r"<time>\s*(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s)?\s*</time>\s*[,:]?\s*([^\n<]+)"
    for match in re.finditer(pattern, think_content, re.I):
        start, end = float(match.group(1)), float(match.group(2))
        caption = match.group(3).strip().rstrip(".,;:")
        if caption and start < end:
            captions.append({"start": start, "end": end, "caption": caption})
    return captions


def calculate_iou(gt_windows: List[List[float]], pred_windows: List[List[float]]) -> float:
    def merge_intervals(intervals: List[List[float]]) -> List[List[float]]:
        valid = [[s, e] for s, e in intervals if s < e]
        if not valid:
            return []
        sorted_intervals = sorted(valid, key=lambda x: x[0])
        merged = [sorted_intervals[0][:]]
        for current in sorted_intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current[:])
        return merged

    all_gt = merge_intervals(gt_windows)
    all_pred = merge_intervals(pred_windows)
    if not all_gt and not all_pred:
        return 1.0
    if not all_gt or not all_pred:
        return 0.0

    total_gt = sum(e - s for s, e in all_gt)
    total_pred = sum(e - s for s, e in all_pred)
    intersection = 0.0
    i = j = 0
    while i < len(all_gt) and j < len(all_pred):
        gt_start, gt_end = all_gt[i]
        pred_start, pred_end = all_pred[j]
        intersection += max(0.0, min(gt_end, pred_end) - max(gt_start, pred_start))
        if gt_end < pred_end:
            i += 1
        else:
            j += 1
    union = total_gt + total_pred - intersection
    return intersection / union if union > 0 else 0.0


def compute_tiou_reward(solution_str: str, ground_truth: str) -> Tuple[float, List, List]:
    pred_windows = extract_time_intervals(solution_str, only_result=True)
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    if not pred_windows and not gt_windows:
        return 1.0, pred_windows, gt_windows
    if not pred_windows or not gt_windows:
        return 0.0, pred_windows, gt_windows
    return calculate_iou(gt_windows, pred_windows), pred_windows, gt_windows


def temporal_iou_single(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    pred_start, pred_end = pred
    gt_start, gt_end = gt
    if pred_start >= pred_end or gt_start >= gt_end:
        return 0.0
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return intersection / union if union > 0 else 0.0


def recall_at_tiou_threshold(
    pred_intervals: List[Tuple[float, float]],
    gt_intervals: List[Tuple[float, float]],
    threshold: float = 0.5,
) -> float:
    if not gt_intervals or not pred_intervals:
        return 0.0
    matched_gt = set()
    for i, gt in enumerate(gt_intervals):
        for pred in pred_intervals:
            if temporal_iou_single(pred, gt) >= threshold:
                matched_gt.add(i)
                break
    return len(matched_gt) / len(gt_intervals)


def compute_recall_reward(
    solution_str: str,
    ground_truth: str,
    threshold: float | None = None,
) -> Tuple[float, List, List]:
    threshold = threshold if threshold is not None else RECALL_TIOU_THRESHOLD
    pred_windows = extract_time_intervals(solution_str, only_result=True)
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    pred_intervals = [(p[0], p[1]) for p in pred_windows]
    gt_intervals = [(g[0], g[1]) for g in gt_windows]
    if not gt_intervals or not pred_intervals:
        return 0.0, pred_intervals, gt_intervals
    return recall_at_tiou_threshold(pred_intervals, gt_intervals, threshold), pred_intervals, gt_intervals


def compute_prf_metrics(
    pred_segments: List[Tuple[float, float]],
    gt_segments: List[Tuple[float, float]],
    iou_thresholds: List[float] | None = None,
) -> Dict[str, float]:
    iou_thresholds = iou_thresholds or PRF_THRESHOLDS
    results: Dict[str, float] = {}
    if not pred_segments or not gt_segments:
        value = 1.0 if not pred_segments and not gt_segments else 0.0
        for th in iou_thresholds:
            results[f"P@{th}"] = value
            results[f"R@{th}"] = value
            results[f"F1@{th}"] = value
        results["precision_avg"] = value
        results["recall_avg"] = value
        results["f1_avg"] = value
        return results

    iou_matrix = np.array([[temporal_iou_single(p, g) for g in gt_segments] for p in pred_segments])
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
    matched_ious = [iou_matrix[i, j] for i, j in zip(pred_indices, gt_indices)]

    precisions = []
    recalls = []
    f1s = []
    for th in iou_thresholds:
        tp = sum(1 for iou in matched_ious if iou >= th)
        precision = tp / len(pred_segments) if pred_segments else 0.0
        recall = tp / len(gt_segments) if gt_segments else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        results[f"P@{th}"] = precision
        results[f"R@{th}"] = recall
        results[f"F1@{th}"] = f1
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    results["precision_avg"] = sum(precisions) / len(precisions)
    results["recall_avg"] = sum(recalls) / len(recalls)
    results["f1_avg"] = sum(f1s) / len(f1s)
    return results


def compute_precision_reward(solution_str: str, ground_truth: str) -> Tuple[float, Dict[str, float]]:
    metrics = compute_prf_metrics(
        [(p[0], p[1]) for p in extract_time_intervals(solution_str, only_result=True)],
        [(g[0], g[1]) for g in extract_time_intervals(ground_truth, only_result=True)],
    )
    return metrics["precision_avg"], metrics


def compute_recall_avg_reward(solution_str: str, ground_truth: str) -> Tuple[float, Dict[str, float]]:
    metrics = compute_prf_metrics(
        [(p[0], p[1]) for p in extract_time_intervals(solution_str, only_result=True)],
        [(g[0], g[1]) for g in extract_time_intervals(ground_truth, only_result=True)],
    )
    return metrics["recall_avg"], metrics


def compute_f1_reward(solution_str: str, ground_truth: str) -> Tuple[float, Dict[str, float]]:
    metrics = compute_prf_metrics(
        [(p[0], p[1]) for p in extract_time_intervals(solution_str, only_result=True)],
        [(g[0], g[1]) for g in extract_time_intervals(ground_truth, only_result=True)],
    )
    return metrics["f1_avg"], metrics


def compute_prf_all_rewards(solution_str: str, ground_truth: str) -> Tuple[float, float, float, Dict[str, float]]:
    metrics = compute_prf_metrics(
        [(p[0], p[1]) for p in extract_time_intervals(solution_str, only_result=True)],
        [(g[0], g[1]) for g in extract_time_intervals(ground_truth, only_result=True)],
    )
    return metrics["precision_avg"], metrics["recall_avg"], metrics["f1_avg"], metrics


def compute_cacc_reward(solution_str: str, ground_truth: str) -> Tuple[float, int, int]:
    pred_count = len(extract_time_intervals(solution_str, only_result=True))
    gt_count = len(extract_time_intervals(ground_truth, only_result=True))
    return (1.0 if pred_count == gt_count else 0.0), pred_count, gt_count


def _is_valid_time_tag(text: str) -> bool:
    return bool(
        re.match(
            r"^<time>\s*\d+(?:\.\d+)?\s*[-–—~]\s*\d+(?:\.\d+)?\s*(?:seconds?|s)?\s*</time>$",
            text.strip(),
            re.I,
        )
    )


def _extract_all_time_tags(text: str) -> List[str]:
    return re.findall(r"<time>.*?</time>", text, re.I | re.DOTALL)


def _validate_time_tags_format(time_tags: List[str]) -> bool:
    return bool(time_tags) and all(_is_valid_time_tag(tag) for tag in time_tags)


def compute_format_reward(input_string: str) -> Tuple[float, Dict[str, Any]]:
    pred_dict: Dict[str, Any] = {}
    input_string = input_string.strip()
    if not input_string:
        pred_dict["error"] = "empty_response"
        return 0.0, pred_dict

    think_open_match = re.search(r"<think>", input_string, re.I)
    think_close_match = re.search(r"</think>", input_string, re.I)
    has_think_open = think_open_match is not None
    has_think_close = think_close_match is not None

    if has_think_open or has_think_close:
        if not (has_think_open and has_think_close):
            pred_dict["error"] = "mismatched_think_tags"
            return 0.0, pred_dict
        if think_open_match.start() >= think_close_match.start():
            pred_dict["error"] = "think_tags_wrong_order"
            return 0.0, pred_dict
        if _extract_all_time_tags(input_string[: think_open_match.start()]):
            pred_dict["error"] = "timestamps_before_think"
            return 0.0, pred_dict

        think_content = input_string[think_open_match.end() : think_close_match.start()]
        after_think = input_string[think_close_match.end() :]
        pred_dict["think_content"] = think_content.strip()
        pred_dict["after_think"] = after_think.strip()
        think_time_tags = _extract_all_time_tags(think_content)
        pred_dict["think_time_tags_count"] = len(think_time_tags)
        if think_time_tags and not _validate_time_tags_format(think_time_tags):
            pred_dict["error"] = "invalid_think_timestamp_format"
            return 0.0, pred_dict

        result_time_tags = _extract_all_time_tags(after_think)
        pred_dict["result_time_tags_count"] = len(result_time_tags)
        if not result_time_tags:
            pred_dict["error"] = "no_result_timestamp"
            return 0.0, pred_dict
        if not _validate_time_tags_format(result_time_tags):
            pred_dict["error"] = "invalid_result_timestamp_format"
            return 0.0, pred_dict
        result_intervals = extract_time_intervals(after_think, only_result=False)
        if result_intervals:
            pred_dict["result_timestamp"] = result_intervals[0]
        pred_dict["format_type"] = "with_think"
        return 1.0, pred_dict

    pred_dict["format_type"] = "no_think"
    all_time_tags = _extract_all_time_tags(input_string)
    pred_dict["time_tags_count"] = len(all_time_tags)
    if not all_time_tags:
        pred_dict["error"] = "no_timestamp"
        return 0.0, pred_dict
    if not _validate_time_tags_format(all_time_tags):
        pred_dict["error"] = "invalid_timestamp_format"
        return 0.0, pred_dict
    result_intervals = extract_time_intervals(input_string, only_result=False)
    if result_intervals:
        pred_dict["result_timestamp"] = result_intervals[0]
    return 1.0, pred_dict


def _soft_overlong_penalty(length: int, soft: int, hard: int, penalty_factor: float) -> float:
    if length <= soft:
        return 0.0
    if length <= hard:
        return penalty_factor * ((length - soft) / (hard - soft))
    return penalty_factor


def compute_length_penalty(think_content: str, captions: List[Dict[str, Any]]) -> float:
    think_penalty = _soft_overlong_penalty(
        len(think_content) if think_content else 0,
        THINK_LENGTH_SOFT_THRESHOLD,
        THINK_LENGTH_HARD_THRESHOLD,
        THINK_OVERLONG_PENALTY_FACTOR,
    )
    caption_penalty = 0.0
    if captions:
        penalties = [
            _soft_overlong_penalty(
                len(c.get("caption", "")),
                CAPTION_LENGTH_SOFT_THRESHOLD,
                CAPTION_LENGTH_HARD_THRESHOLD,
                CAPTION_OVERLONG_PENALTY_FACTOR,
            )
            for c in captions
        ]
        caption_penalty = sum(penalties) / len(penalties)
    return think_penalty + caption_penalty


STAGE1_PROMPT_WITH_GT = """You are a STRICT evaluator for Video Temporal Grounding caption quality.

## Context
Query: "{query}"
Ground Truth: {num_gt_intervals} segment(s) at {gt_intervals_str}
Video duration: ~{video_duration:.0f}s

## Model's Captions:
{caption_list_str}

## EVALUATION TASK

### Step 1: Map each GT to captions
For each GT segment, find the BEST matching caption (if any).
A match requires: (1) temporal overlap, AND (2) caption describes "{query}"

### Step 2: Score STRICTLY using these rules

**COVERAGE (0-10)**: What fraction of GT segments are matched?
- 10 = ALL {num_gt_intervals} GT matched with clear "{query}" descriptions
- 8 = ALL matched, but 1 has weak description
- 6 = ~70% matched
- 4 = ~50% matched
- 2 = Only 1 matched
- 0 = None matched
If ANY GT is missing, score <= 8.

**PRECISION (0-10)**: How close are boundaries?
- 10 = ALL within 1s of GT
- 8 = Most within 2s
- 6 = Within 3-5s
- 4 = Off by 5-10s
- 2 = Off by >10s
Captions much WIDER than GT count as imprecise.

**DISCRIMINABILITY (0-10)**: Can occurrences be distinguished?
- 10 = Each has unique context
- 7 = Good context for most
- 4 = Generic descriptions
- 0 = Impossible to distinguish

Output ONLY valid JSON:
{{"coverage": <int 0-10>, "precision": <int 0-10>, "discriminability": <int 0-10>}}"""


STAGE2_PROMPT_NO_GT = """You are predicting video timestamps from text captions ONLY.

## Query: "{query}"

## Captions:
{caption_list_str}

## Task
Find ALL segments where "{query}" occurs based on the captions.

## Rules:
1. Look for captions that DESCRIBE or IMPLY "{query}"
2. Use the caption timestamp as prediction
3. If multiple captions match, list all of them
4. Output one segment per line as "start - end"

## Example Output:
10.5 - 15.0
32.0 - 37.0"""


def _parse_stage1_response(response_text: str) -> Dict[str, int]:
    json_match = re.search(r"```(?:json)?\s*(\{[^`]+\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return {
                "coverage": min(10, max(0, int(data.get("coverage", 0)))),
                "precision": min(10, max(0, int(data.get("precision", 0)))),
                "discriminability": min(10, max(0, int(data.get("discriminability", 0)))),
            }
        except Exception:
            pass

    json_match = re.search(r'\{[^{}]*"coverage"[^{}]*\}', response_text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return {
                "coverage": min(10, max(0, int(data.get("coverage", 0)))),
                "precision": min(10, max(0, int(data.get("precision", 0)))),
                "discriminability": min(10, max(0, int(data.get("discriminability", 0)))),
            }
        except Exception:
            pass

    result = {"coverage": 0, "precision": 0, "discriminability": 0}
    for key in result:
        match = re.search(rf'{key}["\s:]+(\d+)', response_text, re.I)
        if match:
            result[key] = min(10, int(match.group(1)))
    return result


def _parse_stage2_response_multi(response_text: str) -> List[Tuple[float, float]]:
    lower = response_text.lower()
    if len(response_text) < 30 and ("none" in lower or "no " in lower or "cannot" in lower):
        return []
    intervals = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)", response_text):
        start, end = float(match.group(1)), float(match.group(2))
        if 0 <= start < end <= 10000:
            intervals.append((start, end))
    return intervals


def compute_caption_reward(solution_str: str, ground_truth: str, extra_info: Dict[str, Any]) -> Tuple[float, int, List[float]]:
    coverage_score = 0.0
    precision_score = 0.0
    discriminability_score = 0.0
    counterfactual_score = 0.0

    think_content = extract_think_content(solution_str)
    if not think_content:
        return 0.0, 0, [0.0, 0.0, 0.0, 0.0]
    captions = extract_captions_with_timestamps(think_content)
    if not captions:
        return 0.0, 0, [0.0, 0.0, 0.0, 0.0]

    query = extra_info.get("query", "")
    if not query:
        return 0.0, len(captions), [0.0, 0.0, 0.0, 0.0]
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    if not gt_windows:
        return 0.0, len(captions), [0.0, 0.0, 0.0, 0.0]

    num_gt_intervals = len(gt_windows)
    gt_intervals_str = ", ".join([f"[{g[0]:.1f}s - {g[1]:.1f}s]" for g in gt_windows])
    video_duration = max(max(c["end"] for c in captions), max(g[1] for g in gt_windows) + 5.0)
    caption_list_str = "\n".join([f"[{c['start']:.1f}s - {c['end']:.1f}s]: {c['caption']}" for c in captions])

    try:
        llm = _get_judge_client()
    except Exception:
        count_ratio = min(len(captions), num_gt_intervals) / max(len(captions), num_gt_intervals, 1)
        return 0.4 * count_ratio, len(captions), [5.0, 5.0, 5.0, 0.0]

    stage1_prompt = STAGE1_PROMPT_WITH_GT.format(
        query=query,
        num_gt_intervals=num_gt_intervals,
        gt_intervals_str=gt_intervals_str,
        video_duration=video_duration,
        caption_list_str=caption_list_str,
    )
    for retry in range(MAX_RETRIES):
        try:
            response = llm.chat([{"role": "user", "content": stage1_prompt}])
            parsed = _parse_stage1_response(response.content)
            if any(v > 0 for v in parsed.values()):
                coverage_score = parsed["coverage"] / 10.0
                precision_score = parsed["precision"] / 10.0
                discriminability_score = parsed["discriminability"] / 10.0
                if debug_print:
                    print(
                        f"[S1] cov={parsed['coverage']}, "
                        f"prec={parsed['precision']}, disc={parsed['discriminability']}"
                    )
                break
        except Exception as exc:
            if debug_print:
                print(f"[S1 Error] retry {retry + 1}/{MAX_RETRIES}: {exc}")

    stage2_prompt = STAGE2_PROMPT_NO_GT.format(query=query, caption_list_str=caption_list_str)
    for retry in range(MAX_RETRIES):
        try:
            response = llm.chat([{"role": "user", "content": stage2_prompt}])
            judge_predictions = _parse_stage2_response_multi(response.content)
            if judge_predictions:
                gt_segments = [(g[0], g[1]) for g in gt_windows]
                prf_metrics = compute_prf_metrics(judge_predictions, gt_segments, [0.3, 0.5])
                counterfactual_score = (prf_metrics.get("F1@0.3", 0.0) + prf_metrics.get("F1@0.5", 0.0)) / 2.0
                if debug_print:
                    print(
                        f"[S2] preds={len(judge_predictions)}, "
                        f"F1@0.3={prf_metrics.get('F1@0.3', 0):.3f}, "
                        f"F1@0.5={prf_metrics.get('F1@0.5', 0):.3f}"
                    )
                break
            if debug_print:
                print(f"[S2] No predictions from: {response.content[:100]}...")
        except Exception as exc:
            if debug_print:
                print(f"[S2 Error] retry {retry + 1}/{MAX_RETRIES}: {exc}")

    combined_score = (
        CAPTION_COVERAGE_WEIGHT * coverage_score
        + CAPTION_PRECISION_WEIGHT * precision_score
        + CAPTION_DISCRIMINABILITY_WEIGHT * discriminability_score
        + CAPTION_COUNTERFACTUAL_WEIGHT * counterfactual_score
    )
    raw_scores = [
        coverage_score * 10,
        precision_score * 10,
        discriminability_score * 10,
        counterfactual_score * 10,
    ]
    return combined_score, len(captions), raw_scores


def tg_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    del data_source
    if extra_info is None:
        extra_info = {}

    strategy = get_current_strategy()
    enabled_rewards = strategy.get_enabled_rewards()
    weights = strategy.get_weights()
    result: Dict[str, Any] = {"score": 0.0, "strategy_name": strategy.get_name()}

    try:
        if solution_str is None:
            solution_str = ""
        if not solution_str.strip():
            return result

        tiou_score = 0.0
        format_score = 0.0
        caption_score = 0.0
        recall_score = 0.0
        precision_score = 0.0
        f1_score = 0.0
        cacc_score = 0.0
        length_penalty = 0.0

        if "tiou" in enabled_rewards:
            tiou_score, _, _ = compute_tiou_reward(solution_str, ground_truth)
            result["tiou_score"] = tiou_score

        if "format" in enabled_rewards:
            format_score, _ = compute_format_reward(solution_str)
            result["format_score"] = format_score

        if "recall" in enabled_rewards and "recall_avg" not in enabled_rewards:
            recall_score, _, _ = compute_recall_reward(solution_str, ground_truth)
            result["recall_score"] = recall_score

        if any(r in enabled_rewards for r in ["precision", "recall_avg", "f1"]):
            precision_score, recall_score, f1_score, prf_metrics = compute_prf_all_rewards(solution_str, ground_truth)
            if "precision" in enabled_rewards:
                result["precision_score"] = precision_score
                for th in PRF_THRESHOLDS:
                    result[f"P@{th}"] = prf_metrics[f"P@{th}"]
            if "recall_avg" in enabled_rewards:
                result["recall_score"] = recall_score
                for th in PRF_THRESHOLDS:
                    result[f"R@{th}"] = prf_metrics[f"R@{th}"]
            if "f1" in enabled_rewards:
                result["f1_score"] = f1_score
                for th in PRF_THRESHOLDS:
                    result[f"F1@{th}"] = prf_metrics[f"F1@{th}"]

        if "cacc" in enabled_rewards:
            cacc_score, pred_count, gt_count = compute_cacc_reward(solution_str, ground_truth)
            result["cacc_score"] = cacc_score
            result["pred_count"] = pred_count
            result["gt_count"] = gt_count

        if "caption" in enabled_rewards:
            think_content = extract_think_content(solution_str)
            captions = extract_captions_with_timestamps(think_content) if think_content else []
            caption_score, caption_count, raw_scores = compute_caption_reward(solution_str, ground_truth, extra_info)
            result["caption_score"] = caption_score
            result["caption_count"] = caption_count if caption_count > 0 else len(captions)
            result["caption_raw_scores"] = raw_scores
            if captions:
                total_caption_length = sum(len(c.get("caption", "")) for c in captions)
                result["caption_avg_length"] = total_caption_length / len(captions)
            else:
                result["caption_avg_length"] = 0.0

        if "length_penalty" in enabled_rewards:
            think_content = extract_think_content(solution_str)
            captions = extract_captions_with_timestamps(think_content) if think_content else []
            length_penalty = compute_length_penalty(think_content, captions)
            result["length_penalty"] = length_penalty

        combined_score = (
            weights.tiou * tiou_score
            + weights.format * format_score
            + weights.caption * caption_score
            + weights.recall * recall_score
            + weights.precision * precision_score
            + weights.f1 * f1_score
            + weights.cacc * cacc_score
            - weights.length_penalty * length_penalty
        )
        result["score"] = max(0.0, combined_score)
        return result

    except Exception:
        print(f"[ERROR] Exception in tg_reward: {traceback.format_exc()}")
        return result
