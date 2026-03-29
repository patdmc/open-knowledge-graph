"""
Differentiable Cox partial likelihood loss for survival prediction.
"""

import torch
import torch.nn as nn


class CoxPartialLikelihoodLoss(nn.Module):
    """Negative log partial likelihood for Cox proportional hazards.

    For each uncensored event at time t_i:
        L_i = hazard_i - log(sum_{j in R(t_i)} exp(hazard_j))

    where R(t_i) is the risk set at time t_i (all patients alive at t_i).

    Uses the Breslow approximation: sort by descending time, compute
    cumulative log-sum-exp for efficient risk set calculation.
    """

    def forward(self, hazard, time, event):
        """
        Args:
            hazard: (N,) predicted log partial hazard for each patient
            time: (N,) survival time
            event: (N,) event indicator (1=death, 0=censored)

        Returns:
            Scalar negative log partial likelihood (lower is better).
        """
        # Sort by descending time
        order = torch.argsort(time, descending=True)
        hazard = hazard[order]
        event = event[order]

        # Cumulative log-sum-exp over risk set (Efron approx not needed here)
        # After sorting descending, risk set for patient i = patients 0..i
        log_risk = torch.logcumsumexp(hazard, dim=0)

        # Log partial likelihood: sum over uncensored events
        uncensored = event.bool()
        if uncensored.sum() == 0:
            return torch.tensor(0.0, device=hazard.device, requires_grad=True)

        log_lik = hazard[uncensored] - log_risk[uncensored]
        return -log_lik.mean()


class StratifiedCoxLoss(nn.Module):
    """Cox partial likelihood computed WITHIN treatment strata.

    Forces the model to learn biology (discriminate patients on the same
    treatment) rather than treatment selection (sicker patients get more
    aggressive treatment). Computes Cox loss separately within each stratum
    and averages, weighted by number of events.
    """

    def __init__(self):
        super().__init__()
        self.cox = CoxPartialLikelihoodLoss()

    def forward(self, hazard, time, event, strata):
        """
        Args:
            hazard: (N,) predicted log partial hazard
            time: (N,) survival time
            event: (N,) event indicator (1=death, 0=censored)
            strata: (N,) long — stratum index per patient (e.g., treatment arm)
        """
        unique_strata = strata.unique()
        total_loss = torch.tensor(0.0, device=hazard.device, requires_grad=True)
        total_events = 0

        for s in unique_strata:
            mask = strata == s
            s_event = event[mask]
            n_events = s_event.sum().item()
            if n_events < 2:
                continue

            s_loss = self.cox(hazard[mask], time[mask], s_event)
            total_loss = total_loss + s_loss * n_events
            total_events += n_events

        if total_events == 0:
            return torch.tensor(0.0, device=hazard.device, requires_grad=True)

        return total_loss / total_events
