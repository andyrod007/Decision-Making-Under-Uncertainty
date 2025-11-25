# Andres Rodriguez
# 005304109

import pandas as pd
import streamlit as st
from decision_rules import (
    summarize_all, expected_payoff, minimax_regret, laplace, maximin, maximax
)

st.set_page_config(page_title="Decision Making Under Uncertainty", layout="wide")

# Welcome form
st.title("Decision Making Under Uncertainty")
st.write("Use this tool to compare alternatives under uncertain events using classic decision rules")

with st.expander("Instructions", expanded=False):
    st.markdown(
        """
        **Steps**
        1. Enter the number of events(n) and alternatives(m).
        2. Fill the payoff table (rows = alternatives, columns = events). 
        3. (Optional) Enter event likelihoods if known.
        4. Choose a selection rule in **Selection Rules** and view the best alternative.
        5. See **Reports** for the regret table and a consolidated summary. 
        """
    )

# Data Analysis Menu
st.header("Data Analysis")
cols = st.columns(3)
with cols[0]:
    m = st.number_input("Total number of alternatives (m)", min_value=1, value=3, step=1)
with cols[1]:
    n = st.number_input("Total number of events(n)", min_value=1, value=3, step=1)
with cols[2]:
    use_example = st.toggle("Load example data", value=False, help="Preload the small demo matrix.")

# defaults
default_alt_labels = [f"Alt {i+1}" for i in range(m)]
default_event_labels = [f"Event {j+1}" for j in range(n)]
if use_example and m == 3 and n == 3:
    pay_df = pd.DataFrame(
        [[8, 4, 12], [6, 10, 5], [9, 7, 9]],
        columns=["Low", "Medium", "High"],
        index=["A", "B", "C"]
    )
    alt_labels = list(pay_df.index)
    event_labels = list(pay_df.columns)
else:
    pay_df = pd.DataFrame(0.0, index=default_alt_labels, columns=default_event_labels)
    alt_labels = default_alt_labels
    event_labels = default_event_labels

st.subheader("Payoff Table (m x n)")
st.caption("Enter the payoff for each alternative (row) under each event (column).")
pay_df = st.data_editor(pay_df, use_container_width=True, num_rows="dynamic")

# Label editors
with st.expander("Rename alternatives and events"):
    alt_labels = st.data_editor(pd.DataFrame({"Alternatives": pay_df.index.tolist()}),
                                use_container_width=True)["Alternatives"].tolist()
    event_labels = st.data_editor(pd.DataFrame({"Events": pay_df.columns.tolist()}),
                                  use_container_width=True)["Events"].tolist()
    # Add labels
    pay_df.index = alt_labels
    pay_df.columns = event_labels

# Likelihoods option
st.subheader("Enter the Likelihood of Events (optional)")
st.caption("If provided, probabilities are normalized to sum to 1. Leave blank to skip.")
prob_cols = st.columns([3, 1])
with prob_cols[0]:
    probs_df = pd.DataFrame([[None]*len(event_labels)], columns=event_labels, index=["Probability"])
    probs_df = st.data_editor(probs_df, use_container_width=True, num_rows=1, key="probs_editor")
with prob_cols[1]:
    normalize = st.checkbox("Normalize", value=True, help="Scale probabilities to sum to 1 if needed.")

# Exact numeric payoffs
try:
    payoffs = pay_df.astype(float).values.tolist()
except Exception as e:
    st.error(f"Please ensure the payoff table contains only numeric values. Details: {e}")
    st.stop()

# Exact probabilities if provided
probs_list = None
if probs_df is not None:
    raw = probs_df.iloc[0].tolist()
    if any(x is not None for x in raw):
        try:
            probs_list = [float(x) if x is not None else 0.0 for x in raw]
            if normalize:
                s = sum(max(0.0, p) for p in probs_list)
                if s <= 0:
                    st.warning("All provided probabilities are non-positive; ignoring.")
                    probs_list = None
                else:
                    probs_list = [max(0.0, p) / s for p in probs_list]
        except Exception as e:
            st.warning(f"Could not parse probabilities: ignoring. Details: {e}")
            probs_list = None

# Selection rules
st.header("Selection Rules")
rule = st.selectbox(
    "Choose a decision rule",
    ["maximin", "maximax", "Laplace (equal likelihoods)", "minimax regret", "expected payoff"],
    index=0
)

run = st.button("Apply Rule")
if run:
    try:
        if rule == "maximin":
            idx, val = maximin(payoffs)
            st.success(f"Best Alternative (maximin): **{alt_labels[idx]}** with value **{val:.4f}**")
        elif rule == "maximax":
            idx, val = maximax(payoffs)
            st.success(f"Best Alternative (maximax): **{alt_labels[idx]}** with value **{val:.4f}**")
        elif rule == "Laplace (equal likelihoods)":
            idx, val = laplace(payoffs)
            st.success(f"Best Alternative (Laplace): **{alt_labels[idx]}** with average **{val:.4f}**")
        elif rule == "minimax regret":
            idx, val, R = minimax_regret(payoffs)
            st.success(f"Best Alternative (minimax regret): **{alt_labels[idx]}** with worst regret **{val:.4f}**")
            # Regret Table
            r_df = pd.DataFrame(R, index=default_event_labels, columns=default_alt_labels)
            st.subheader("Regret Table")
            st.dataframe(r_df, use_container_width=True)
        elif rule == "expected payoff":
            if probs_list is None:
                st.error("Expected payoff requires event likelihoods. Please provide probabilities above.")
            else:
                idx, val = expected_payoff(payoffs, probs_list)
                st.success(f"Best alternative (expected payoff): **{alt_labels[idx]}** with expected value **{val:.4f}**")
    except Exception as e:
        st.error(f"Error while applying rule: {e}")

# Reports
st.header("Reports")
report = summarize_all(payoffs, probs_list, alt_labels, event_labels)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Best Alternatives by Rule")
    rows = []
    for key in ["maximin", "maximax", "laplace", "minimax regret"]:
        label = report[key]["label"]
        value = report[key]["value"]
        rows.append([key, label, value])
    if "expected_payoff" in report:
        rows.append(["expected_payoff", report["expected_payoff"]["label"], report["expected_payoff"]["value"]])
    rep_df = pd.DataFrame(rows, columns=["Rule", "Best Alternative", "Value"])
    st.dataframe(rep_df, use_container_width=True)

with c2:
    st.subheader("Regret Table")
    r_vals = report["regret_table"]["values"]
    r_df = pd.DataFrame(r_vals, index=report["regret_table"]["rows"], columns=report["regret_table"]["cols"])
    st.dataframe(r_df, use_container_width=True, height=300)

st.info("Tip: Use the **Download** button in the three-dot menu on top each to export data.")
