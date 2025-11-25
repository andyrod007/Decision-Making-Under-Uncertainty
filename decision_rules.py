# Andres Rodriguez
# 005304109

from typing import List, Tuple, Optional

Matrix = List[List[float]]

def _validate_payoffs(payoffs: Matrix) -> Tuple[int, int]:
    if not payoffs or not isinstance(payoffs, list):
        raise ValueError("Payoffs must be a non-empty of lists.")
    m = len(payoffs)
    n = len(payoffs[0])
    for row in payoffs:
        if len(row) != n:
            raise ValueError("All rows in payoffs must have the same length.")
    return m, n

def _argmax(values: List[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])

def _argmin(values: List[float]) -> int:
    return min(range(len(values)), key=lambda i: values[i])

def _normalize_probs(probs: List[float]) -> List[float]:
    if any(p < 0 for p in probs):
        raise ValueError("Probabilities must be non-negative.")
    s = sum(probs)
    if s <= 0:
        raise ValueError("Sum of probabilities must be positive.")
    return [p / s for p in probs]

def regret_table(payoffs: Matrix) -> Matrix:
    _, n = _validate_payoffs(payoffs)
    best_by_col = []
    for j in range(n):
        best_by_col.append(max(payoffs[i][j] for i in range(len(payoffs))))
    regrets = [[best_by_col[j] - payoffs[i][j] for j in range(n)] for i in range(len(payoffs))]
    return regrets

def maximin(payoffs: Matrix) -> Tuple[int, float]:
    _validate_payoffs(payoffs)
    worst_per_alt = [min(row) for row in payoffs]
    i_star = _argmax(worst_per_alt)
    return i_star, worst_per_alt[i_star]

def maximax(payoffs: Matrix) -> Tuple[int, float]:
    _validate_payoffs(payoffs)
    best_per_alt = [min(row) for row in payoffs]
    i_star = _argmax(best_per_alt)
    return i_star, best_per_alt[i_star]

def laplace(payoffs: Matrix) -> Tuple[int, float]:
    m, n = _validate_payoffs(payoffs)
    weights = [1.0 / n] * n
    avgs = [sum(payoffs[i][j] * weights[j] for j in range(n)) for i in range(m)]
    i_star = _argmax(avgs)
    return i_star, avgs[i_star]

def minimax_regret(payoffs: Matrix) -> Tuple[int, float, Matrix]:
    _validate_payoffs(payoffs)
    R = regret_table(payoffs)
    max_regret_per_alt = [max(row) for row in R]
    i_star = _argmin(max_regret_per_alt)
    return i_star, max_regret_per_alt[i_star], R

def expected_payoff(payoffs: Matrix, probs: List[float]) -> Tuple[int, float]:
    m, n = _validate_payoffs(payoffs)
    if len(probs) != n:
        raise ValueError(f"Length of the probs ({len(probs)}) must equal number of events ({n}).")
    p = _normalize_probs(probs)
    exp_values = [sum(payoffs[i][j] * p[j] for j in range(n)) for i in range(m)]
    i_star = _argmax(exp_values)
    return i_star, exp_values[i_star]

def summarize_all(payoffs: Matrix, probs: Optional[List[float]] = None,
                  alt_labels: Optional[List[str]] = None,
                  event_labels: Optional[List[str]] = None) -> dict:
    m, n = _validate_payoffs(payoffs)
    alt_labels = alt_labels or [f"Alt {i+1}" for i in range(m)]
    event_labels = event_labels or [f"Event {j+1}" for j in range(n)]

    i_maximin, v_maximin = maximin(payoffs)
    i_maximax, v_maximax = maximax(payoffs)
    i_laplace, v_laplace = laplace(payoffs)
    i_minimax_regret, v_minimax_regret, R = minimax_regret(payoffs)

    summary = {
        "maximin": {"index": i_maximin, "label": alt_labels[i_maximin], "value": v_maximin},
        "maximax": {"index": i_maximax, "label": alt_labels[i_maximax], "value": v_maximax},
        "laplace": {"index": i_laplace, "label": alt_labels[i_laplace], "value": v_laplace},
        "minimax regret": {"index": i_minimax_regret, "label": alt_labels[i_minimax_regret], "value": v_minimax_regret},
        "regret_table": {
            "rows": alt_labels,
            "cols": event_labels,
            "values": R
        }
    }

    if probs is not None:
        i_exp, v_exp, = expected_payoff(payoffs, probs)
        summary["expected payoff"] = {"index": i_exp, "label": alt_labels[i_exp], "value": v_exp}
    return summary

if __name__ == "__main__":
    payoffs = [
        [8, 4, 12],
        [6, 10, 5],
        [9, 7, 9]
    ]
    probs = [0.3, 0.5, 0.2]
    alts = ["A", "B", "C"]
    events = ["Low", "Medium", "High"]

    out = summarize_all(payoffs, probs, alts, events)
    from pprint import pprint
    pprint(out)
