import concurrent.futures
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from omtg_reward.tg_reward import tg_reward
from omtg_utils.log_utils import logger

MAX_REWARD_WORKERS = 32


def batch_tg_reward_fn(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
    valid_response_lengths: Optional[List[int]] = None,
    max_workers: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    del valid_response_lengths, kwargs

    n_samples = len(data_sources)
    if extra_infos is None:
        extra_infos = [None] * n_samples
    max_workers = min(n_samples, max_workers or MAX_REWARD_WORKERS)

    def process_one(args):
        idx, data_source, solution_str, ground_truth, extra_info = args
        reward_dict = tg_reward(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return idx, reward_dict

    args_list = [
        (i, data_sources[i], solution_strs[i], ground_truths[i], extra_infos[i])
        for i in range(n_samples)
    ]

    results: List[Optional[Dict[str, Any]]] = [None] * n_samples
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, args): args[0] for args in args_list}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result_idx, result = future.result()
                results[result_idx] = result
            except Exception as exc:
                logger.error("Future failed for item %s: %s", idx, exc)
                results[idx] = {"score": 0.0, "strategy_name": "error"}

    return results
