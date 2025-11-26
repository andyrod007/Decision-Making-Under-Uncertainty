[README.txt](https://github.com/user-attachments/files/23778165/README.txt)
# Decision Making Under Uncertainty (Python + Streamlit)

## Files
- `decision_rules.py`: Reusable Python module containing the decision rules.
- `app.py`: Streamlit app implementing the full UI (welcome form, data entry, selection rules, reports).

## Quick Start
```bash
pip install streamlit pandas
streamlit run app.py
```

## Example (module only)
```python
from decision_rules import summarize_all

payoffs = [
    [8,  4, 12],   # Alt A
    [6, 10,  5],   # Alt B
    [9,  7,  9]    # Alt C
]
probs = [0.3, 0.5, 0.2]
alts = ["A", "B", "C"]
events = ["Low", "Medium", "High"]

summary = summarize_all(payoffs, probs, alts, events)
print(summary)
```
