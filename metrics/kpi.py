# metrics/kpi.py

import numpy as np

def compute_episode_kpis(
    J_EM,
    J_P,
    violations_any,
    P_rx_off,
    P_p_max,
    eps=1e-6,
):
    """
    Compute KPIs for a single episode.

    Parameters
    ----------
    J_EM : np.ndarray
        EM metric time series (P_rx,on).
    J_P : np.ndarray
        Plasma power metric time series.
    violations_any : np.ndarray of ints (0/1)
        1 if any constraint violated at that time step, else 0.
    P_rx_off : float
        Reference EM power for plasma-off case.
    P_p_max : float
        Max allowed plasma power (for normalization).
    eps : float
        Small number to avoid division by zero.

    Returns
    -------
    kpis : dict
        {
          "avg_J_EM",
          "avg_J_P",
          "avg_delta_EM_dB",
          "eta_EM",
          "R_safe",
        }
    """

    J_EM = np.asarray(J_EM, dtype=float)
    J_P = np.asarray(J_P, dtype=float)
    violations_any = np.asarray(violations_any, dtype=float)

    # --- Basic averages ---
    avg_J_EM = float(np.mean(J_EM))
    avg_J_P = float(np.mean(J_P))

    # --- Clamp EM metric to avoid log10 of non-positive values ---
    # physically this corresponds to "very low but nonzero" EM
    J_EM_clamped = np.maximum(J_EM, 0.01 * P_rx_off)

    delta_EM_dB = 10.0 * np.log10(P_rx_off / J_EM_clamped)
    avg_delta_EM_dB = float(np.mean(delta_EM_dB))

    # --- EM-per-power efficiency (dB per unit power) ---
    if avg_J_P > eps:
        eta_EM = avg_delta_EM_dB / avg_J_P
    else:
        eta_EM = 0.0

    # --- Safe operation ratio ---
    # 1 - fraction of time steps with any violation
    R_safe = float(1.0 - np.mean(violations_any > 0.5))

    kpis = {
        "avg_J_EM": avg_J_EM,
        "avg_J_P": avg_J_P,
        "avg_delta_EM_dB": avg_delta_EM_dB,
        "eta_EM": eta_EM,
        "R_safe": R_safe,
    }
    return kpis


def summarize_kpis(kpi_list):
    """
    Given a list of KPI dicts (one per episode), compute
    mean / std / min / max for each KPI.

    Returns
    -------
    summary : dict[str, dict[str, float]]
        e.g. summary["avg_delta_EM_dB"]["mean"]
    """
    if len(kpi_list) == 0:
        raise ValueError("kpi_list is empty.")

    keys = list(kpi_list[0].keys())
    summary = {}

    for key in keys:
        vals = np.array([k[key] for k in kpi_list], dtype=float)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return summary
