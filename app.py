# import streamlit as st
# import pandas as pd
# import numpy as np
# import altair as alt
# import datetime

# st.set_page_config(layout="wide")
# st.title("Payment Routing Bandit Evaluation")

# # --- Description Box ---
# with st.expander("ℹ️ **How to Use This App**", expanded=True):
#     st.markdown("""
# **This dashboard simulates and evaluates various multi-armed bandit algorithms for online payment gateway routing.**

# **What does it do?**
# - Loads payment data from a CSV.
# - Lets you filter by currency, payment method, payment channel, and gateway.
# - Splits data into time intervals (by day, hour, week, etc).
# - For each interval, computes recommended routing proportions using your chosen algorithm (Thompson Sampling, UCB, Epsilon-Greedy, BGE).
# - Shows *counterfactual* success rates: "If you routed X% to gateway A and Y% to gateway B, what would be the expected success rate?"
# - Compares with BBRv2 offline evaluation and the 'oracle' best possible for that interval.
# - Plots both success rate and regret for each algorithm.

# **How to read:**
# 1. Use the sidebar to choose filters and time grouping.
# 2. Select your preferred routing algorithm. The default is Thompson Sampling (recommended).
# 3. See for each interval: the routing policy, number of transactions, success rates, and regret compared to the best possible choice.
# 4. Visualizations show how routing recommendations and performance change over time.
# """)

# # --- Load Data from CSV ---
# @st.cache_data(show_spinner="Loading dataset...")
# def load_data():
#     df_sim = pd.read_csv("dataset_gateway_routing.csv")
#     # Adapt these column names if your csv is different!
#     df_sim['date'] = pd.to_datetime(df_sim['request_created_time']).dt.date
#     df_sim['datetime'] = pd.to_datetime(df_sim['request_created_time'])
#     return df_sim

# df_sim = load_data()

# # --- Sidebar: Filter & Interval Selection ---
# st.sidebar.header("Filter & Interval")

# currency_choices = sorted(df_sim['currency'].dropna().unique())
# pm_choices = sorted(df_sim['payment_method'].dropna().unique())
# pc_choices = sorted(df_sim['payment_channel'].dropna().unique())
# gw_choices = sorted(df_sim['gateway'].dropna().unique())

# default_currency = ['IDR'] if 'IDR' in currency_choices else currency_choices
# default_pm = ['QR_CODE'] if 'QR_CODE' in pm_choices else pm_choices
# default_pc = ['QRIS'] if 'QRIS' in pc_choices else pc_choices
# default_gw = gw_choices

# interval_choice = st.sidebar.selectbox(
#     "Aggregation Interval",
#     [
#         'Per Day', 'Per 6 Hours', 'Per 4 Hours', 'Per 2 Hours', 'Per 1 Hour',
#         'Per Week', 'Per Month'
#     ], index=0
# )

# # --- Create timegroup BEFORE using min_t/max_t ---
# if interval_choice == 'Per Day':
#     df_sim['timegroup'] = df_sim['datetime'].dt.floor('D')
# elif interval_choice == 'Per 6 Hours':
#     df_sim['timegroup'] = df_sim['datetime'].dt.floor('6H')
# elif interval_choice == 'Per 4 Hours':
#     df_sim['timegroup'] = df_sim['datetime'].dt.floor('4H')
# elif interval_choice == 'Per 2 Hours':
#     df_sim['timegroup'] = df_sim['datetime'].dt.floor('2H')
# elif interval_choice == 'Per 1 Hour':
#     df_sim['timegroup'] = df_sim['datetime'].dt.floor('H')
# elif interval_choice == 'Per Week':
#     df_sim['timegroup'] = df_sim['datetime'].dt.to_period('W').dt.start_time
# elif interval_choice == 'Per Month':
#     df_sim['timegroup'] = df_sim['datetime'].dt.to_period('M').dt.start_time

# min_t = df_sim['timegroup'].min().date()
# max_t = df_sim['timegroup'].max().date()
# # --- Default range: 2025-07-01 to now or max in data
# default_start = datetime.date(2025, 7, 1) if min_t <= datetime.date(2025,7,1) <= max_t else min_t
# default_end = max_t

# currency = st.sidebar.multiselect("Currency", currency_choices, default=default_currency)
# pm = st.sidebar.multiselect("Payment Method", pm_choices, default=default_pm)
# pc = st.sidebar.multiselect("Payment Channel", pc_choices, default=default_pc)
# gw = st.sidebar.multiselect("Gateway", gw_choices, default=default_gw)
# start_date, end_date = st.sidebar.date_input(
#     "Date range",
#     [default_start, default_end],
#     min_value=min_t,
#     max_value=max_t
# )

# # --- Filter Data ---
# df_view = df_sim.copy()
# if currency: df_view = df_view[df_view['currency'].isin(currency)]
# if pm: df_view = df_view[df_view['payment_method'].isin(pm)]
# if pc: df_view = df_view[df_view['payment_channel'].isin(pc)]
# if gw: df_view = df_view[df_view['gateway'].isin(gw)]
# df_view = df_view[(df_view['timegroup'].dt.date >= start_date) & (df_view['timegroup'].dt.date <= end_date)]

# st.write("Historical Dataset:")
# st.dataframe(df_view, use_container_width=True)
# csv_data = df_view.to_csv(index=False)
# st.download_button("Download Dataset", csv_data, "filtered_payments.csv")

# # --- Algorithm Choice ---
# st.sidebar.header("Routing Algorithm")
# algos = ["Thompson Sampling (Bayesian)", "UCB", "Epsilon-Greedy", "Boltzmann-Gumbel"]
# algo_choice = st.sidebar.selectbox("Choose Algorithm", algos, index=0)
# epsilon = st.sidebar.slider("Epsilon (Epsilon-Greedy only)", min_value=0.01, max_value=0.5, value=0.2, step=0.01)
# C_bge = st.sidebar.slider("C (B-Gumbel only)", min_value=0.01, max_value=3.0, value=1.0, step=0.01)
# alpha_softmax = st.sidebar.slider("Softmax Alpha", min_value=1, max_value=30, value=10, step=1)

# # --- Rolling Bandit Processing (by algorithm) ---
# st.header("Bandit Evaluation (Counterfactual, BBRv2, Regret)")
# with st.spinner('Processing...'):
#     groupby_keys = ['payment_method', 'payment_channel']
#     timegroups = sorted(df_view['timegroup'].unique())
#     gateway_list = sorted(df_view['gateway'].unique())
#     n_gw = len(gateway_list)
#     results = []
#     for (payment_method, payment_channel), group in df_view.groupby(groupby_keys):
#         group = group.sort_values('timegroup')
#         counts_dict = {algo: {'success': np.ones(n_gw), 'failure': np.ones(n_gw), 'total': np.zeros(n_gw)} for algo in algos}
#         for i in range(len(timegroups)-1):
#             history_data = group[group['timegroup'].isin(timegroups[:i+1])]
#             # --- Routing probabilities ---
#             # Thompson
#             samples = np.random.beta(counts_dict['Thompson Sampling (Bayesian)']['success'], counts_dict['Thompson Sampling (Bayesian)']['failure'])
#             prop_ts = np.zeros(n_gw)
#             prop_ts[np.argmax(samples)] = 1.0
#             # UCB
#             total_ucb = counts_dict['UCB']['success'] + counts_dict['UCB']['failure']
#             mean_ucb = counts_dict['UCB']['success'] / (total_ucb + 1e-6)
#             ucb = mean_ucb + np.sqrt(2 * np.log(i+2) / (total_ucb + 1e-6))
#             prop_ucb = np.zeros(n_gw)
#             prop_ucb[np.argmax(ucb)] = 1.0
#             # Epsilon-Greedy
#             mean_eg = counts_dict['Epsilon-Greedy']['success'] / (counts_dict['Epsilon-Greedy']['success'] + counts_dict['Epsilon-Greedy']['failure'] + 1e-6)
#             prop_eg = np.ones(n_gw) * epsilon / n_gw
#             prop_eg[np.argmax(mean_eg)] += 1 - epsilon
#             # Boltzmann-Gumbel
#             mean_bge = counts_dict['Boltzmann-Gumbel']['success'] / (counts_dict['Boltzmann-Gumbel']['success'] + counts_dict['Boltzmann-Gumbel']['failure'] + 1e-6)
#             N_bge = counts_dict['Boltzmann-Gumbel']['success'] + counts_dict['Boltzmann-Gumbel']['failure'] + 1e-6
#             beta = C_bge / np.sqrt(N_bge)
#             gumbel = np.random.gumbel(size=n_gw)
#             bge_score = mean_bge + beta * gumbel
#             prop_bge = np.exp(bge_score / alpha_softmax)
#             prop_bge /= prop_bge.sum()
#             # Softmax (for Counterfactual & BBRv2 eval only, not for actual routing)
#             success_counts = 1 + np.array([(history_data['gateway'] == gw).sum() for gw in gateway_list])
#             failure_counts = 1 + np.array([(history_data[(history_data['gateway'] == gw) & (history_data['status'] != 'SUCCESS')].shape[0]) for gw in gateway_list])
#             prob_success = success_counts / (success_counts + failure_counts)
#             exp_probs = np.exp(alpha_softmax * prob_success)
#             prop_softmax = exp_probs / exp_probs.sum()
#             # Choose
#             if algo_choice == "Thompson Sampling (Bayesian)": props = prop_ts
#             elif algo_choice == "UCB": props = prop_ucb
#             elif algo_choice == "Epsilon-Greedy": props = prop_eg
#             elif algo_choice == "Boltzmann-Gumbel": props = prop_bge
#             prop_dict = {gw: float(p) for gw, p in zip(gateway_list, props)}
#             next_time = timegroups[i+1]
#             test_data = group[group['timegroup']==next_time].copy()
#             n_tx = len(test_data)
#             n_per_gateway = (props * n_tx).round().astype(int)
#             allocation = []
#             for gw, n in zip(gateway_list, n_per_gateway):
#                 allocation += [gw] * n
#             if len(allocation) < n_tx:
#                 allocation += [gateway_list[np.argmax(props)]] * (n_tx - len(allocation))
#             elif len(allocation) > n_tx:
#                 allocation = allocation[:n_tx]
#             np.random.shuffle(allocation)
#             test_data['assigned_gateway'] = allocation
#             alloc_tx_dict = {gw: int((test_data['assigned_gateway'] == gw).sum()) for gw in gateway_list}
#             cf_success_rate_dict = {}
#             for gw in gateway_list:
#                 n_gw_ = (test_data['gateway'] == gw).sum()
#                 n_success = ((test_data['gateway'] == gw) & (test_data['status'] == 'SUCCESS')).sum()
#                 cf_success_rate_dict[gw] = float(n_success / n_gw_) if n_gw_ > 0 else np.nan
#             # Counterfactual: expected success rate
#             success_rate_cf = sum(
#                 prop_dict[gw] * cf_success_rate_dict[gw]
#                 for gw in gateway_list if not np.isnan(cf_success_rate_dict[gw])
#             )
#             # BBRv2
#             behavior_prop = test_data['gateway'].value_counts(normalize=True).to_dict()
#             behavior_prop = {gw: behavior_prop.get(gw, 0.0) for gw in gateway_list}
#             bbrv2_weighted_sum = 0.0
#             bbrv2_sum_weight = 0.0
#             bbrv2_sr_per_gateway = {}
#             for gw in gateway_list:
#                 mask = (test_data['gateway'] == gw)
#                 n = mask.sum()
#                 if n == 0 or behavior_prop[gw] == 0:
#                     bbrv2_sr_per_gateway[gw] = np.nan
#                     continue
#                 rewards = (test_data.loc[mask, 'status'] == 'SUCCESS').astype(float)
#                 policy_prob = prop_dict[gw]
#                 behavior_prob = behavior_prop[gw]
#                 bbrv2_sr_per_gateway[gw] = (policy_prob / behavior_prob) * rewards.mean()
#                 bbrv2_weighted_sum += ((policy_prob / behavior_prob) * rewards).sum()
#                 bbrv2_sum_weight += n
#             bbrv2_success_rate = bbrv2_weighted_sum / bbrv2_sum_weight if bbrv2_sum_weight > 0 else np.nan
#             # Oracle
#             best_gw = max(
#                 cf_success_rate_dict,
#                 key=lambda k: cf_success_rate_dict[k] if not np.isnan(cf_success_rate_dict[k]) else -1
#             )
#             oracle_success_rate = cf_success_rate_dict[best_gw] if best_gw else np.nan
#             oracle_gateway = best_gw if best_gw else np.nan
#             # Regret tracking
#             regret_cf = oracle_success_rate - success_rate_cf if n_tx else np.nan
#             regret_bbrv2 = oracle_success_rate - bbrv2_success_rate if n_tx else np.nan
#             results.append({
#                 'interval': next_time,
#                 'payment_method': payment_method,
#                 'payment_channel': payment_channel,
#                 'softmax_prop': prop_dict,
#                 'alloc_tx_dict': alloc_tx_dict,
#                 'cf_success_rate_dict': cf_success_rate_dict,
#                 'bbrv2_sr_per_gateway': bbrv2_sr_per_gateway,
#                 'success_rate_counterfactual': success_rate_cf,
#                 'success_rate_bbrv2': bbrv2_success_rate,
#                 'oracle_success_rate': oracle_success_rate,
#                 'oracle_gateway': oracle_gateway,
#                 'regret_counterfactual': regret_cf,
#                 'regret_bbrv2': regret_bbrv2,
#                 'total_tx': n_tx,
#                 'algo_used': algo_choice,
#             })
#             # Update counts for next round (simulate learning)
#             for algo in algos:
#                 if algo == "Thompson Sampling (Bayesian)":
#                     chosen = np.argmax(samples)
#                     assigned = test_data['assigned_gateway'].values
#                     observed = (test_data['assigned_gateway'] == test_data['gateway']) & (test_data['status'] == 'SUCCESS')
#                     counts_dict[algo]['success'][chosen] += observed.sum()
#                     counts_dict[algo]['failure'][chosen] += (assigned == gateway_list[chosen]).sum() - observed.sum()
#                 elif algo == "UCB":
#                     chosen = np.argmax(ucb)
#                     assigned = test_data['assigned_gateway'].values
#                     observed = (test_data['assigned_gateway'] == test_data['gateway']) & (test_data['status'] == 'SUCCESS')
#                     counts_dict[algo]['success'][chosen] += observed.sum()
#                     counts_dict[algo]['failure'][chosen] += (assigned == gateway_list[chosen]).sum() - observed.sum()
#                 elif algo == "Epsilon-Greedy":
#                     chosen = np.argmax(mean_eg)
#                     assigned = test_data['assigned_gateway'].values
#                     observed = (test_data['assigned_gateway'] == test_data['gateway']) & (test_data['status'] == 'SUCCESS')
#                     counts_dict[algo]['success'][chosen] += observed.sum()
#                     counts_dict[algo]['failure'][chosen] += (assigned == gateway_list[chosen]).sum() - observed.sum()
#                 elif algo == "Boltzmann-Gumbel":
#                     chosen = np.argmax(bge_score)
#                     assigned = test_data['assigned_gateway'].values
#                     observed = (test_data['assigned_gateway'] == test_data['gateway']) & (test_data['status'] == 'SUCCESS')
#                     counts_dict[algo]['success'][chosen] += observed.sum()
#                     counts_dict[algo]['failure'][chosen] += (assigned == gateway_list[chosen]).sum() - observed.sum()
#     # Dataframe formatting
#     all_long_rows = []
#     for row in results:
#         interval = row['interval']
#         total_tx = row['total_tx']
#         payment_method = row['payment_method']
#         payment_channel = row['payment_channel']
#         success_rate_counterfactual = row['success_rate_counterfactual']
#         success_rate_bbrv2 = row['success_rate_bbrv2']
#         oracle_success_rate = row['oracle_success_rate']
#         oracle_gateway = row['oracle_gateway']
#         regret_counterfactual = row['regret_counterfactual']
#         regret_bbrv2 = row['regret_bbrv2']
#         algo_used = row['algo_used']
#         for gw in gateway_list:
#             all_long_rows.append({
#                 'interval': interval,
#                 'payment_method': payment_method,
#                 'payment_channel': payment_channel,
#                 'gateway': gw,
#                 'prop': row['softmax_prop'].get(gw, np.nan),
#                 'alloc': row['alloc_tx_dict'].get(gw, np.nan),
#                 'cf_sr': row['cf_success_rate_dict'].get(gw, np.nan),
#                 'bbrv2_sr': row['bbrv2_sr_per_gateway'].get(gw, np.nan),
#                 'success_rate_counterfactual': success_rate_counterfactual,
#                 'success_rate_bbrv2': success_rate_bbrv2,
#                 'oracle_success_rate': oracle_success_rate,
#                 'oracle_gateway': oracle_gateway,
#                 'regret_counterfactual': regret_counterfactual,
#                 'regret_bbrv2': regret_bbrv2,
#                 'total_tx': total_tx,
#                 'algo_used': algo_used,
#             })
#     df_long = pd.DataFrame(all_long_rows)
#     st.success(f"Analysis done. {df_long.shape[0]} rows")

#     st.subheader("Result Table (Newest First, Filterable)")
#     df_long = df_long.sort_values("interval", ascending=False)

#     # Option 1: Use Streamlit Data Editor for interactive filtering/sorting
#     st.data_editor(
#         df_long,
#         use_container_width=True,
#         num_rows="dynamic",
#         column_order=list(df_long.columns),
#         hide_index=True,
#         key="result_editor"
#     )

#     csv = df_long.to_csv(index=False)
#     st.download_button("Download CSV", csv, "bandit_eval.csv")


#     # Visualization
#     st.subheader("Visualization")
#     selected_gws = st.multiselect(
#         "Select Gateway(s) to plot",
#         sorted(df_long['gateway'].unique()),
#         default=sorted(df_long['gateway'].unique()),
#         key="viz_gateway"
#     )
#     df_plot = df_long[df_long['gateway'].isin(selected_gws)]
#     # Success rate (Counterfactual)
#     st.markdown("**Counterfactual Success Rate (per Gateway per Interval)**")
#     st.altair_chart(
#         alt.Chart(df_plot).mark_line(point=True).encode(
#             x='interval:T',
#             y='success_rate_counterfactual:Q',
#             color='gateway:N',
#             tooltip=['interval', 'gateway', 'success_rate_counterfactual', 'success_rate_bbrv2', 'cf_sr', 'bbrv2_sr']
#         ).properties(title='Counterfactual Success Rate per Gateway over Time'),
#         use_container_width=True
#     )
#     # Success rate (BBRv2)
#     st.markdown("**BBRv2 Success Rate (per Gateway per Interval)**")
#     st.altair_chart(
#         alt.Chart(df_plot).mark_line(point=True).encode(
#             x='interval:T',
#             y='success_rate_bbrv2:Q',
#             color='gateway:N',
#             tooltip=['interval', 'gateway', 'success_rate_bbrv2', 'success_rate_counterfactual', 'cf_sr', 'bbrv2_sr']
#         ).properties(title='BBRv2 Success Rate per Gateway over Time'),
#         use_container_width=True
#     )

#     # --- Regret Visualization ---
#     st.markdown("**Regret of Your Chosen Algorithm**")
#     df_regret = df_long.drop_duplicates(['interval', 'payment_method', 'payment_channel'])  # One row per interval
#     st.altair_chart(
#         alt.Chart(df_regret).mark_line(point=True).encode(
#             x='interval:T',
#             y='regret_counterfactual:Q',
#             color='algo_used:N',
#             tooltip=['interval', 'regret_counterfactual', 'success_rate_counterfactual', 'oracle_success_rate', 'oracle_gateway']
#         ).properties(title='Regret (Counterfactual) Over Time'),
#         use_container_width=True
#     )

#     st.markdown("""
#     ---
#     **How to interpret:**  
#     - *Success Rate Counterfactual* shows the expected gateway success rate if you followed the routing policy for each interval, **using historical data**.  
#     - *BBRv2 Success Rate* is a more unbiased offline estimate, adjusting for the actual distribution in the data.
#     - *Regret* = Difference between your success rate and the *oracle* (best possible if you always picked the winner gateway that interval).
#     - *Zero Regret* means your routing is as good as the best possible on that interval!
#     - Choose different algorithms from the sidebar to compare their regret and policy behavior.
#     """)



import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime

st.set_page_config(layout="wide")
st.title("Payment Routing Bandit Evaluation")

# --- Description Box ---
with st.expander("ℹ️ **How to Use This App**", expanded=True):
    st.markdown("""
**This dashboard simulates and evaluates various multi-armed bandit algorithms for online payment gateway routing.**

**What does it do?**
- Loads payment data from a CSV.
- Lets you filter by currency, payment method, payment channel, and gateway.
- Splits data into time intervals (by day, hour, week, etc).
- For each interval, computes recommended routing proportions using your chosen algorithm (Thompson Sampling, UCB, Epsilon-Greedy, BGE).
- Shows *counterfactual* success rates: "If you routed X% to gateway A and Y% to gateway B, what would be the expected success rate?"
- Compares with the 'oracle' best possible for that interval.
- Plots both success rate and regret for each algorithm.

**How to read:**
1. Use the sidebar to choose filters and time grouping.
2. Select your preferred routing algorithm. The default is Thompson Sampling (recommended).
3. See for each interval: the routing policy, number of transactions, success rates, and regret compared to the best possible choice.
4. Visualizations show how routing recommendations and performance change over time.
""")

# --- Load Data from CSV ---
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df_sim = pd.read_csv("dataset_gateway_routing.csv")
    # Adapt these column names if your csv is different!
    df_sim['date'] = pd.to_datetime(df_sim['request_created_time']).dt.date
    df_sim['datetime'] = pd.to_datetime(df_sim['request_created_time'])
    return df_sim

df_sim = load_data()

# --- Sidebar: Filter & Interval Selection ---
st.sidebar.header("Filter & Interval")

currency_choices = sorted(df_sim['currency'].dropna().unique())
pm_choices = sorted(df_sim['payment_method'].dropna().unique())
pc_choices = sorted(df_sim['payment_channel'].dropna().unique())
gw_choices = sorted(df_sim['gateway'].dropna().unique())

default_currency = ['IDR'] if 'IDR' in currency_choices else currency_choices
default_pm = ['QR_CODE'] if 'QR_CODE' in pm_choices else pm_choices
default_pc = ['QRIS'] if 'QRIS' in pc_choices else pc_choices
default_gw = gw_choices

interval_choice = st.sidebar.selectbox(
    "Aggregation Interval",
    [
        'Per Day', 'Per 6 Hours', 'Per 4 Hours', 'Per 2 Hours', 'Per 1 Hour',
        'Per Week', 'Per Month'
    ], index=0
)

# --- Create timegroup BEFORE using min_t/max_t ---
if interval_choice == 'Per Day':
    df_sim['timegroup'] = df_sim['datetime'].dt.floor('D')
elif interval_choice == 'Per 6 Hours':
    df_sim['timegroup'] = df_sim['datetime'].dt.floor('6H')
elif interval_choice == 'Per 4 Hours':
    df_sim['timegroup'] = df_sim['datetime'].dt.floor('4H')
elif interval_choice == 'Per 2 Hours':
    df_sim['timegroup'] = df_sim['datetime'].dt.floor('2H')
elif interval_choice == 'Per 1 Hour':
    df_sim['timegroup'] = df_sim['datetime'].dt.floor('H')
elif interval_choice == 'Per Week':
    df_sim['timegroup'] = df_sim['datetime'].dt.to_period('W').dt.start_time
elif interval_choice == 'Per Month':
    df_sim['timegroup'] = df_sim['datetime'].dt.to_period('M').dt.start_time

min_t = df_sim['timegroup'].min().date()
max_t = df_sim['timegroup'].max().date()
# --- Default range: 2025-07-01 to now or max in data
default_start = datetime.date(2025, 7, 1) if min_t <= datetime.date(2025,7,1) <= max_t else min_t
default_end = max_t

currency = st.sidebar.multiselect("Currency", currency_choices, default=default_currency)
pm = st.sidebar.multiselect("Payment Method", pm_choices, default=default_pm)
pc = st.sidebar.multiselect("Payment Channel", pc_choices, default=default_pc)
gw = st.sidebar.multiselect("Gateway", gw_choices, default=default_gw)
start_date, end_date = st.sidebar.date_input(
    "Date range",
    [default_start, default_end],
    min_value=min_t,
    max_value=max_t
)

# --- Filter Data ---
df_view = df_sim.copy()
if currency: df_view = df_view[df_view['currency'].isin(currency)]
if pm: df_view = df_view[df_view['payment_method'].isin(pm)]
if pc: df_view = df_view[df_view['payment_channel'].isin(pc)]
if gw: df_view = df_view[df_view['gateway'].isin(gw)]
df_view = df_view[(df_view['timegroup'].dt.date >= start_date) & (df_view['timegroup'].dt.date <= end_date)]

st.write("Historical Dataset:")
st.dataframe(df_view, use_container_width=True)
csv_data = df_view.to_csv(index=False)
st.download_button("Download Dataset", csv_data, "filtered_payments.csv")

# --- Algorithm Choice ---
st.sidebar.header("Routing Algorithm")
algos = ["Thompson Sampling (Bayesian)", "UCB", "Epsilon-Greedy", "Boltzmann-Gumbel"]
algo_choice = st.sidebar.selectbox("Choose Algorithm", algos, index=0)
epsilon = st.sidebar.slider("Epsilon (Epsilon-Greedy only)", min_value=0.01, max_value=0.5, value=0.2, step=0.01)
C_bge = st.sidebar.slider("C (B-Gumbel only)", min_value=0.01, max_value=3.0, value=1.0, step=0.01)
alpha_softmax = st.sidebar.slider("Softmax Alpha", min_value=1, max_value=30, value=10, step=1)

# --- Rolling Bandit Processing (by algorithm) ---
st.header("Bandit Evaluation (Counterfactual, Regret)")
with st.spinner('Processing...'):
    groupby_keys = ['payment_method', 'payment_channel']
    timegroups = sorted(df_view['timegroup'].unique())
    gateway_list = sorted(df_view['gateway'].unique())
    n_gw = len(gateway_list)
    results = []
    for (payment_method, payment_channel), group in df_view.groupby(groupby_keys):
        group = group.sort_values('timegroup')
        counts_dict = {algo: {'success': np.ones(n_gw), 'failure': np.ones(n_gw), 'total': np.zeros(n_gw)} for algo in algos}
        for i in range(len(timegroups)-1):
            history_data = group[group['timegroup'].isin(timegroups[:i+1])]
            # --- Routing probabilities ---
            # Thompson
            samples = np.random.beta(counts_dict['Thompson Sampling (Bayesian)']['success'], counts_dict['Thompson Sampling (Bayesian)']['failure'])
            prop_ts = np.zeros(n_gw)
            prop_ts[np.argmax(samples)] = 1.0
            # UCB
            total_ucb = counts_dict['UCB']['success'] + counts_dict['UCB']['failure']
            mean_ucb = counts_dict['UCB']['success'] / (total_ucb + 1e-6)
            ucb = mean_ucb + np.sqrt(2 * np.log(i+2) / (total_ucb + 1e-6))
            prop_ucb = np.zeros(n_gw)
            prop_ucb[np.argmax(ucb)] = 1.0
            # Epsilon-Greedy
            mean_eg = counts_dict['Epsilon-Greedy']['success'] / (counts_dict['Epsilon-Greedy']['success'] + counts_dict['Epsilon-Greedy']['failure'] + 1e-6)
            prop_eg = np.ones(n_gw) * epsilon / n_gw
            prop_eg[np.argmax(mean_eg)] += 1 - epsilon
            # Boltzmann-Gumbel
            mean_bge = counts_dict['Boltzmann-Gumbel']['success'] / (counts_dict['Boltzmann-Gumbel']['success'] + counts_dict['Boltzmann-Gumbel']['failure'] + 1e-6)
            N_bge = counts_dict['Boltzmann-Gumbel']['success'] + counts_dict['Boltzmann-Gumbel']['failure'] + 1e-6
            beta = C_bge / np.sqrt(N_bge)
            gumbel = np.random.gumbel(size=n_gw)
            bge_score = mean_bge + beta * gumbel
            prop_bge = np.exp(bge_score / alpha_softmax)
            prop_bge /= prop_bge.sum()
            # Choose
            if algo_choice == "Thompson Sampling (Bayesian)": props = prop_ts
            elif algo_choice == "UCB": props = prop_ucb
            elif algo_choice == "Epsilon-Greedy": props = prop_eg
            elif algo_choice == "Boltzmann-Gumbel": props = prop_bge
            prop_dict = {gw: float(p) for gw, p in zip(gateway_list, props)}
            next_time = timegroups[i+1]
            test_data = group[group['timegroup']==next_time].copy()
            n_tx = len(test_data)
            n_per_gateway = (props * n_tx).round().astype(int)
            allocation = []
            for gw, n in zip(gateway_list, n_per_gateway):
                allocation += [gw] * n
            if len(allocation) < n_tx:
                allocation += [gateway_list[np.argmax(props)]] * (n_tx - len(allocation))
            elif len(allocation) > n_tx:
                allocation = allocation[:n_tx]
            np.random.shuffle(allocation)
            test_data['assigned_gateway'] = allocation
            alloc_tx_dict = {gw: int((test_data['assigned_gateway'] == gw).sum()) for gw in gateway_list}
            cf_success_rate_dict = {}
            for gw in gateway_list:
                n_gw_ = (test_data['gateway'] == gw).sum()
                n_success = ((test_data['gateway'] == gw) & (test_data['status'] == 'SUCCESS')).sum()
                cf_success_rate_dict[gw] = float(n_success / n_gw_) if n_gw_ > 0 else np.nan
            # Counterfactual: expected success rate
            success_rate_cf = sum(
                prop_dict[gw] * cf_success_rate_dict[gw]
                for gw in gateway_list if not np.isnan(cf_success_rate_dict[gw])
            )
            # Oracle
            best_gw = max(
                cf_success_rate_dict,
                key=lambda k: cf_success_rate_dict[k] if not np.isnan(cf_success_rate_dict[k]) else -1
            )
            oracle_success_rate = cf_success_rate_dict[best_gw] if best_gw else np.nan
            oracle_gateway = best_gw if best_gw else np.nan
            # Regret tracking
            regret_cf = oracle_success_rate - success_rate_cf if n_tx else np.nan
            results.append({
                'Interval': next_time,
                'Payment Method': payment_method,
                'Payment Channel': payment_channel,
                'Gateway': None,  # Will be set in flattening loop
                'Routing Allocation (%)': None,
                'Routed Transactions (#)': None,
                'Gateway Success Rate (CF)': None,
                'Overall Success Rate (CF)': success_rate_cf,
                'Oracle Best Success Rate': oracle_success_rate,
                'Oracle Gateway': oracle_gateway,
                'Regret (Counterfactual)': regret_cf,
                'Total Transactions': n_tx,
                'Routing Algorithm': algo_choice,
                'Per Gateway Softmax Prop': prop_dict,
                'Per Gateway Alloc': alloc_tx_dict,
                'Per Gateway SR': cf_success_rate_dict
            })
            # Update counts for next round (simulate learning)
            for algo in algos:
                if algo == "Thompson Sampling (Bayesian)":
                    chosen = np.argmax(samples)
                    assigned = test_data['assigned_gateway'].values
                    observed = (test_data['assigned_gateway'] == test_data['gateway']) & (test_data['status'] == 'SUCCESS')
                    counts_dict[algo]['success'][chosen] += observed.sum()
                    counts_dict[algo]['failure'][chosen] += (assigned == gateway_list[chosen]).sum() - observed.sum()
                elif algo == "UCB":
                    chosen = np.argmax(ucb)
                    assigned = test_data['assigned_gateway'].values
                    observed = (test_data['assigned_gateway'] == test_data['gateway']) & (test_data['status'] == 'SUCCESS')
                    counts_dict[algo]['success'][chosen] += observed.sum()
                    counts_dict[algo]['failure'][chosen] += (assigned == gateway_list[chosen]).sum() - observed.sum()
                elif algo == "Epsilon-Greedy":
                    chosen = np.argmax(mean_eg)
                    assigned = test_data['assigned_gateway'].values
                    observed = (test_data['assigned_gateway'] == test_data['gateway']) & (test_data['status'] == 'SUCCESS')
                    counts_dict[algo]['success'][chosen] += observed.sum()
                    counts_dict[algo]['failure'][chosen] += (assigned == gateway_list[chosen]).sum() - observed.sum()
                elif algo == "Boltzmann-Gumbel":
                    chosen = np.argmax(bge_score)
                    assigned = test_data['assigned_gateway'].values
                    observed = (test_data['assigned_gateway'] == test_data['gateway']) & (test_data['status'] == 'SUCCESS')
                    counts_dict[algo]['success'][chosen] += observed.sum()
                    counts_dict[algo]['failure'][chosen] += (assigned == gateway_list[chosen]).sum() - observed.sum()
    # Dataframe formatting (flatten per gateway)
    all_long_rows = []
    for row in results:
        interval = row['Interval']
        total_tx = row['Total Transactions']
        payment_method = row['Payment Method']
        payment_channel = row['Payment Channel']
        overall_sr = row['Overall Success Rate (CF)']
        oracle_sr = row['Oracle Best Success Rate']
        oracle_gateway = row['Oracle Gateway']
        regret = row['Regret (Counterfactual)']
        algo_used = row['Routing Algorithm']
        for gw in gateway_list:
            all_long_rows.append({
                'Interval': interval,
                'Payment Method': payment_method,
                'Payment Channel': payment_channel,
                'Gateway': gw,
                'Routing Allocation (%)': row['Per Gateway Softmax Prop'].get(gw, np.nan),
                'Routed Transactions (#)': row['Per Gateway Alloc'].get(gw, np.nan),
                'Gateway Success Rate (CF)': row['Per Gateway SR'].get(gw, np.nan),
                'Overall Success Rate (CF)': overall_sr,
                'Oracle Best Success Rate': oracle_sr,
                'Oracle Gateway': oracle_gateway,
                'Regret (Counterfactual)': regret,
                'Total Transactions': total_tx,
                'Routing Algorithm': algo_used
            })
    df_long = pd.DataFrame(all_long_rows)

    # Format percentage columns for human-readability
    for col in ['Routing Allocation (%)', 'Gateway Success Rate (CF)', 'Overall Success Rate (CF)', 'Oracle Best Success Rate', 'Regret (Counterfactual)']:
        df_long[col] = (df_long[col] * 100).round(2)

    st.success(f"Analysis done. {df_long.shape[0]} rows")

    st.markdown("""
    ### ℹ️ How to Interpret (Result Table)
    - **Interval**: The time window (e.g., per day, per hour) being evaluated.
    - **Routing Allocation (%)**: What % of transactions would be sent to this gateway *by the chosen algorithm* for this interval.
    - **Routed Transactions (#)**: How many transactions are assigned to this gateway in simulation for that interval.
    - **Gateway Success Rate (CF)**: *Counterfactual success rate* — What % of transactions would have succeeded if routed here (historical, not real-time).
    - **Overall Success Rate (CF)**: Weighted average success rate across all gateways for this interval (using the algorithm’s routing policy).
    - **Oracle Best Success Rate**: The highest possible success rate you could get for this interval (if you always picked the best gateway after seeing the results).
    - **Regret (Counterfactual)**: The difference between your algorithm’s success rate and the oracle’s (lower is better).
    - **Routing Algorithm**: The algorithm used for routing recommendation.

    **Definitions**  
    - *Counterfactual Success Rate*: The expected success rate if, in the past, you had routed transactions according to this algorithm’s recommendation.
    - *Oracle Success Rate*: The best possible (in hindsight) success rate, if you had known which gateway would perform best for that interval.

    **Tips:**
    - Compare different algorithms by switching the selector in the sidebar.
    - Lower regret means your algorithm is close to “perfect” (oracle).
    - Click table headers to sort, or search/filter by column as needed.
    """)

    st.subheader("Result Table (Newest First, Filterable)")
    df_long = df_long.sort_values("Interval", ascending=False)

    st.data_editor(
        df_long,
        use_container_width=True,
        num_rows="dynamic",
        column_order=list(df_long.columns),
        hide_index=True,
        key="result_editor"
    )

    csv = df_long.to_csv(index=False)
    st.download_button("Download CSV", csv, "bandit_eval.csv")

    # Visualization
    st.subheader("Visualization")
    selected_gws = st.multiselect(
        "Select Gateway(s) to plot",
        sorted(df_long['Gateway'].unique()),
        default=sorted(df_long['Gateway'].unique()),
        key="viz_gateway"
    )
    df_plot = df_long[df_long['Gateway'].isin(selected_gws)]
    # Success rate (Counterfactual)
    st.markdown("**Counterfactual Success Rate (per Gateway per Interval)**")
    st.altair_chart(
        alt.Chart(df_plot).mark_line(point=True).encode(
            x='Interval:T',
            y='Gateway Success Rate (CF):Q',
            color='Gateway:N',
            tooltip=['Interval', 'Gateway', 'Gateway Success Rate (CF)', 'Overall Success Rate (CF)']
        ).properties(title='Counterfactual Success Rate per Gateway over Time'),
        use_container_width=True
    )

    # --- Regret Visualization ---
    st.markdown("**Regret of Your Chosen Algorithm**")
    df_regret = df_long.drop_duplicates(['Interval', 'Payment Method', 'Payment Channel'])  # One row per interval
    st.altair_chart(
        alt.Chart(df_regret).mark_line(point=True).encode(
            x='Interval:T',
            y='Regret (Counterfactual):Q',
            color='Routing Algorithm:N',
            tooltip=['Interval', 'Regret (Counterfactual)', 'Overall Success Rate (CF)', 'Oracle Best Success Rate', 'Oracle Gateway']
        ).properties(title='Regret (Counterfactual) Over Time'),
        use_container_width=True
    )

