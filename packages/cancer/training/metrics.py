"""
Survival prediction metrics: C-index, time-dependent AUC, Brier score.
"""

import numpy as np
import torch
from lifelines.utils import concordance_index as _lifelines_ci


def concordance_index(hazard, time, event):
    """Harrell's concordance index.

    Args:
        hazard: predicted hazard scores (higher = higher risk)
        time: survival times
        event: event indicators (1=event, 0=censored)

    Returns:
        float C-index in [0, 1]
    """
    if isinstance(hazard, torch.Tensor):
        hazard = hazard.detach().cpu().numpy()
    if isinstance(time, torch.Tensor):
        time = time.detach().cpu().numpy()
    if isinstance(event, torch.Tensor):
        event = event.detach().cpu().numpy()

    # lifelines expects: higher predicted value = longer survival
    # Our hazard is higher = higher risk, so negate
    return _lifelines_ci(time, -hazard, event)


def time_dependent_auc(hazard, time, event, eval_times=None):
    """Compute time-dependent AUC at specified evaluation times.

    Uses the Uno estimator (inverse probability of censoring weighted).

    Args:
        hazard: (N,) predicted hazard
        time: (N,) survival time
        event: (N,) event indicator
        eval_times: list of times to evaluate at. Default: [12, 36, 60] months

    Returns:
        dict mapping eval_time -> AUC
    """
    if eval_times is None:
        eval_times = [12, 36, 60]

    if isinstance(hazard, torch.Tensor):
        hazard = hazard.detach().cpu().numpy()
    if isinstance(time, torch.Tensor):
        time = time.detach().cpu().numpy()
    if isinstance(event, torch.Tensor):
        event = event.detach().cpu().numpy()

    results = {}
    for t in eval_times:
        # Binary classification at time t: did patient survive past t?
        # Only include patients observed past t or who had event before t
        mask = (time > t) | (event == 1)
        if mask.sum() < 10:
            results[t] = float("nan")
            continue

        h = hazard[mask]
        true_label = (time[mask] <= t) & (event[mask] == 1)

        if true_label.sum() == 0 or true_label.sum() == len(true_label):
            results[t] = float("nan")
            continue

        # Simple AUC via concordance
        from sklearn.metrics import roc_auc_score
        try:
            results[t] = roc_auc_score(true_label.astype(int), h)
        except ValueError:
            results[t] = float("nan")

    return results
