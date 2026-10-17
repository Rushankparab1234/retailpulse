from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# RetailPulse dashboard. Uses native Streamlit components rather than HTML headers.
st.set_page_config(
    page_title="RetailPulse | Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "background": "#0B1120",
    "card": "#151F32",
    "text": "#E5E7EB",
    "muted": "#94A3B8",
    "teal": "#2DD4BF",
    "high": "#FB7185",
    "medium": "#FBBF24",
    "low": "#34D399",
}
RISK_COLORS = {
    "High Risk": COLORS["high"],
    "Medium Risk": COLORS["medium"],
    "Low Risk": COLORS["low"],
}

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
RETENTION_FILE = OUTPUT_DIR / "customer_retention_priorities.csv"
SUMMARY_FILE = OUTPUT_DIR / "customer_risk_summary.csv"


@st.cache_data(show_spinner="Loading RetailPulse customer data...")
def load_retention_data(file_path: str, modified_time: float) -> pd.DataFrame:
    # modified_time changes when the CSV is replaced, invalidating the cache.
    del modified_time
    return pd.read_csv(file_path)


def load_data() -> pd.DataFrame:
    if not RETENTION_FILE.exists():
        st.error(f"Required file not found: `{RETENTION_FILE}`")
        st.info(
            "Run the export cells in `04_Churn_Prediction.ipynb` first. "
            "The notebook should save `customer_retention_priorities.csv` "
            "inside the project's `outputs` folder."
        )
        st.stop()

    try:
        df = load_retention_data(str(RETENTION_FILE), RETENTION_FILE.stat().st_mtime)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        st.error(f"Could not read the retention CSV: {exc}")
        st.stop()

    required = [
        "customer_id",
        "churn_probability",
        "risk_category",
        "historical_monetary",
        "retention_priority_score",
        "retention_priority",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing required columns in retention CSV: {missing}")
        st.write("Columns found:", list(df.columns))
        st.info(
            "Check the final export cell in `04_Churn_Prediction.ipynb` and ensure "
            "it saves the retention-priority dataframe to the expected file."
        )
        st.stop()

    for col in ["churn_probability", "historical_monetary", "retention_priority_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["customer_id"] = df["customer_id"].astype(str)
    df["risk_category"] = df["risk_category"].fillna("Unknown").astype(str)
    df["retention_priority"] = df["retention_priority"].fillna("Unknown").astype(str)
    df = df.dropna(
        subset=["churn_probability", "historical_monetary", "retention_priority_score"]
    ).copy()
    df["churn_probability"] = df["churn_probability"].clip(0, 1)
    return df


retention_df = load_data()


# Sidebar filters
st.sidebar.title("📊 RetailPulse")
st.sidebar.caption("Customer churn and retention intelligence")
st.sidebar.divider()
st.sidebar.subheader("Dashboard filters")

risk_options = sorted(retention_df["risk_category"].unique().tolist())
priority_options = sorted(retention_df["retention_priority"].unique().tolist())
selected_risks = st.sidebar.multiselect(
    "Customer risk category", risk_options, default=risk_options
)
selected_priorities = st.sidebar.multiselect(
    "Retention priority", priority_options, default=priority_options
)
probability_range = st.sidebar.slider(
    "Churn probability range (%)",
    min_value=0,
    max_value=100,
    value=(0, 100),
    help="Filter customers by their predicted churn probability.",
)
search_customer = st.sidebar.text_input("Find customer ID", placeholder="Enter a customer ID")

filtered_df = retention_df[
    retention_df["risk_category"].isin(selected_risks)
    & retention_df["retention_priority"].isin(selected_priorities)
    & retention_df["churn_probability"].between(
        probability_range[0] / 100, probability_range[1] / 100
    )
].copy()
if search_customer.strip():
    filtered_df = filtered_df[
        filtered_df["customer_id"].str.contains(
            search_customer.strip(), case=False, na=False, regex=False
        )
    ]

# Header: native Streamlit elements, no HTML strings.
st.title("🎯 Churn Prediction")
st.subheader("Customer Churn & Retention Intelligence")
st.write(
    "Identify customers at risk of churning, understand their historical value, "
    "and prioritize retention outreach."
)
st.caption(
    "Predicted churn probabilities are model estimates. Historical spending is "
    "descriptive and is not a guaranteed forecast of future revenue."
)
st.divider()

if filtered_df.empty:
    st.warning("No customers match the selected filters. Adjust the sidebar filters.")
    st.stop()

# KPI cards
total_customers = int(filtered_df["customer_id"].nunique())
high_risk_customers = int(
    filtered_df.loc[filtered_df["risk_category"] == "High Risk", "customer_id"].nunique()
)
predicted_churn_customers = int((filtered_df["churn_probability"] >= 0.36).sum())
average_churn_probability = float(filtered_df["churn_probability"].mean())
historical_spending = float(filtered_df["historical_monetary"].sum())
high_priority_customers = int(
    filtered_df.loc[
        filtered_df["retention_priority"].str.lower().eq("high priority"),
        "customer_id"
    ].nunique()
)

k1, k2, k3 = st.columns(3)
k1.metric("Customers in view", f"{total_customers:,}")
k2.metric("High-risk customers", f"{high_risk_customers:,}")
k3.metric("Average churn probability", f"{average_churn_probability:.1%}")
k4, k5, k6 = st.columns(3)
k4.metric("Predicted churn (≥ 36%)", f"{predicted_churn_customers:,}")
k5.metric("High-priority retention", f"{high_priority_customers:,}")
k6.metric("Historical spending", f"{historical_spending:,.2f}")

st.divider()

# Risk distribution and probability distribution
left, right = st.columns(2)
with left:
    st.subheader("Customer risk distribution")
    risk_counts = (
        filtered_df.groupby("risk_category", as_index=False)
        .agg(customers=("customer_id", "nunique"))
    )
    fig_risk = px.pie(
        risk_counts,
        names="risk_category",
        values="customers",
        hole=0.62,
        color="risk_category",
        color_discrete_map=RISK_COLORS,
    )
    fig_risk.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}<br>Customers: %{value:,}<extra></extra>",
    )
    fig_risk.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        legend_title_text="Risk category",
    )
    st.plotly_chart(fig_risk, use_container_width=True)


with right:
    st.subheader("Churn probability distribution")

    fig_probability = px.histogram(
        filtered_df,
        x="churn_probability",
        nbins=40,
        color_discrete_sequence=[COLORS["teal"]],
        labels={
            "churn_probability": "Predicted churn probability"
        },
    )

    fig_probability.update_layout(
        xaxis_tickformat=".0%",
        xaxis_title="Predicted churn probability",
        yaxis_title="Number of customers",
        bargap=0.05,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )

    fig_probability.update_xaxes(
        gridcolor="#334155",
        range=[0, 1],
    )

    fig_probability.update_yaxes(
        gridcolor="#334155",
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True,
    )

    st.caption(
        "This chart displays predicted probabilities for all customers "
        "matching the selected filters."
    )

    # Zoom into customers with very low predicted probabilities.
    low_probability_df = filtered_df[
        filtered_df["churn_probability"] < 0.05
    ].copy()

    st.subheader("Low churn probability: 0–5%")

    if not low_probability_df.empty:
        fig_low_probability = px.histogram(
            low_probability_df,
            x="churn_probability",
            nbins=20,
            color_discrete_sequence=[COLORS["medium"]],
            labels={
                "churn_probability": "Predicted churn probability"
            },
        )

        fig_low_probability.update_layout(
            xaxis_tickformat=".1%",
            xaxis_title="Predicted churn probability",
            yaxis_title="Number of customers",
            bargap=0.05,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        )

        fig_low_probability.update_xaxes(
            gridcolor="#334155",
            range=[0, 0.05],
        )

        fig_low_probability.update_yaxes(
            gridcolor="#334155",
        )

        st.plotly_chart(
            fig_low_probability,
            use_container_width=True,
        )

        st.caption(
            f"{len(low_probability_df):,} customers have predicted "
            "churn probabilities below 5%."
        )
    else:
        st.info("No customers fall below the 5% probability threshold.")

# Historical spending by risk category
st.divider()
st.subheader("Historical spending by customer risk")
spending_summary = (
    filtered_df.groupby("risk_category", as_index=False)
    .agg(
        customers=("customer_id", "nunique"),
        historical_spending=("historical_monetary", "sum"),
        average_historical_spending=("historical_monetary", "mean"),
        average_churn_probability=("churn_probability", "mean"),
    )
)
fig_spending = px.bar(
    spending_summary,
    x="risk_category",
    y="historical_spending",
    color="risk_category",
    color_discrete_map=RISK_COLORS,
    text_auto=",.2f",
    labels={
        "risk_category": "Risk category",
        "historical_spending": "Total historical spending",
    },
)
fig_spending.update_layout(
    showlegend=False,
    margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"]),
)
fig_spending.update_xaxes(gridcolor="#334155")
fig_spending.update_yaxes(gridcolor="#334155")
st.plotly_chart(fig_spending, use_container_width=True)

summary_display = spending_summary.copy()
summary_display["average_churn_probability"] = summary_display[
    "average_churn_probability"
].map(lambda value: f"{value:.1%}")
summary_display["historical_spending"] = summary_display["historical_spending"].map(
    lambda value: f"{value:,.2f}"
)
summary_display["average_historical_spending"] = summary_display[
    "average_historical_spending"
].map(lambda value: f"{value:,.2f}")
st.dataframe(summary_display, use_container_width=True, hide_index=True)


# CUSTOMER RETENTION RECOMMENDATION ENGINE


def recommend_retention_action(row):

    risk = str(row["risk_category"]).strip().lower()
    priority = str(row["retention_priority"]).strip().lower()

    # High-risk customers
    if risk == "high risk":

        if priority == "high priority":
            return (
                "Personalised retention outreach "
                "with a relevant incentive"
            )

        elif priority == "medium priority":
            return ("Targeted re-engagement campaign")

        else:
            return ("Low-cost automated re-engagement message")

    # Medium-risk customers
    elif risk == "medium risk":

        if priority == "high priority":
            return ("Personalised follow-up and loyalty benefits")

        elif priority == "medium priority":
            return ("Relevant offer or product recommendations")

        else:
            return ("Automated customer engagement campaign")

    # Low-risk customers
    elif risk == "low risk":

        if priority == "high priority":
            return ("Loyalty rewards and exclusive benefits")

        elif priority == "medium priority":
            return ("Cross-selling and complementary products")

        else:
            return ("Standard customer engagement")

    # Fallback for unexpected category values
    return ("Review customer profile before selecting an action")


# Customer retention priority table
st.divider()
st.subheader("Customer retention prioritization")
st.write(
    "Review customers with high churn probability and high retention priority. "
    "The table is sorted by retention priority score, highest first."
)

display_columns = [
    "customer_id",
    "churn_probability",
    "risk_category",
    "historical_monetary",
    "retention_priority_score",
    "retention_priority",
    "recommended_action",
]
for optional_column in ["top_churn_reasons", "top_churn_contributions"]:
    if optional_column in filtered_df.columns:
        display_columns.append(optional_column)

priority_table = filtered_df.sort_values(
    ["retention_priority_score", "churn_probability"], ascending=[False, False]
).copy()
# Generate a retention recommendation for every customer
priority_table["recommended_action"] = priority_table.apply(
    recommend_retention_action,
    axis=1,
)
st.dataframe(
    priority_table[display_columns],
    use_container_width=True,
    hide_index=True,
    column_config={
        "customer_id": st.column_config.TextColumn("Customer ID"),
        "churn_probability": st.column_config.NumberColumn(
            "Churn probability", format="percent"
        ),
        "historical_monetary": st.column_config.NumberColumn(
            "Historical spending", format="%,.2f"
        ),
        "retention_priority_score": st.column_config.NumberColumn(
            "Retention priority score", format="%.2f"
        ),
        "risk_category": st.column_config.TextColumn("Risk category"),
        "retention_priority": st.column_config.TextColumn("Retention priority"),
        "recommended_action": st.column_config.TextColumn("Recommended Action",help="Suggested retention action based on customer risk and priority.",width="large",),
    },
)

# CUSTOMER DETAIL VIEW

st.divider()
st.subheader("Customer Profile")

customer_ids = priority_table["customer_id"].astype(str).tolist()

selected_customer_id = st.selectbox(
    "Select a customer to inspect",
    options=customer_ids,
    key="selected_customer_profile",
)

customer_row = priority_table[
    priority_table["customer_id"].astype(str) == selected_customer_id
].iloc[0]

# Customer metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "Churn Probability",
    f"{customer_row['churn_probability']:.2%}",
)

col2.metric(
    "Historical Spending",
    f"{customer_row['historical_monetary']:,.2f}",
)

col3.metric(
    "Retention Priority Score",
    f"{customer_row['retention_priority_score']:.2f}",
)

# Risk and priority
col4, col5 = st.columns(2)

col4.info(f"Risk Category: {customer_row['risk_category']}")

col5.info(f"Retention Priority: {customer_row['retention_priority']}")

# Recommended action
st.markdown("#### Recommended Retention Action")

st.success(customer_row["recommended_action"])

# SHAP explanations, if available

# CUSTOMER CHURN EXPLANATIONS WITH SHAP CONTRIBUTIONS

import ast

st.markdown("#### Why was this customer flagged?")

# SHAP feature names and their business-friendly labels
reason_labels = {
    "recency_days": "Time since last purchase",
    "frequency": "Purchase frequency",
    "monetary": "Historical monetary value",
    "total_quantity": "Total quantity purchased",
    "average_order_value": "Average order value",
    "distinct_products": "Product variety",
    "avg_days_between_purchases": "Average time between purchases",
    "median_days_between_purchases": "Typical time between purchases",
    "active_months": "Active purchasing months",
    "recent_frequency": "Recent purchase frequency",
    "recent_monetary": "Recent spending",
    "earlier_frequency": "Earlier purchase frequency",
    "earlier_monetary": "Earlier spending",
    "earlier_frequency_60d": "Earlier purchase frequency (60-day window)",
    "earlier_monetary_60d": "Earlier spending (60-day window)",
    "spending_change": "Change in spending",
    "frequency_change": "Change in purchase frequency",
    "has_repeat_purchase": "Repeat purchase indicator",
    "observed_window_span_days": "Observed purchasing history duration",
    "purchase_interval_count": "Number of purchase intervals",
    "missingindicator_avg_days_between_purchases":
        "Missing average purchase interval",
    "missingindicator_median_days_between_purchases":
        "Missing median purchase interval",
}


# Get the customer's signed SHAP contributions from the profile row.
# This assumes the feature columns in the exported retention CSV
# contain the signed SHAP values, as in your notebook output.

contribution_columns = [
    feature
    for feature in reason_labels
    if feature in customer_row.index
]

contributions = pd.Series(
    {
        feature: pd.to_numeric(
            customer_row[feature],
            errors="coerce",
        )
        for feature in contribution_columns
    },
    dtype="float64",
).dropna()

# Keep only features with nonzero contributions.
contributions = contributions[contributions != 0]

if contributions.empty:

    st.info(
        "Signed SHAP contributions are not available in the "
        "current retention report for this customer."
    )

else:

    positive_contributions = (
        contributions[contributions > 0]
        .sort_values(ascending=False)
        .head(3)
    )

    negative_contributions = (
        contributions[contributions < 0]
        .sort_values(ascending=True)
        .head(3)
    )

    left, right = st.columns(2)

    with left:

        st.markdown("##### Factors pushing toward churn")

        if positive_contributions.empty:

            st.write("No positive contributions were found.")

        else:

            for feature, value in positive_contributions.items():

                label = reason_labels.get(
                    feature,
                    feature.replace("_", " ").capitalize(),
                )

                st.markdown(
                    f"**{label}**  \n"
                    f"SHAP contribution: `+{value:.4f}`"
                )

    with right:

        st.markdown("##### Factors pushing away from churn")

        if negative_contributions.empty:

            st.write("No negative contributions were found.")

        else:

            for feature, value in negative_contributions.items():

                label = reason_labels.get(
                    feature,
                    feature.replace("_", " ").capitalize(),
                )

                st.markdown(
                    f"**{label}**  \n"
                    f"SHAP contribution: `{value:.4f}`"
                )

    # Display the strongest contributions in a chart.
    chart_contributions = pd.concat(
        [
            positive_contributions,
            negative_contributions,
        ]
    ).sort_values()

    if not chart_contributions.empty:

        chart_df = pd.DataFrame({
            "Feature": [
                reason_labels.get(
                    feature,
                    feature.replace("_", " ").capitalize(),
                )
                for feature in chart_contributions.index
            ],
            "SHAP contribution": chart_contributions.values,
        })

        import plotly.express as px

        chart_df["Direction"] = chart_df["SHAP contribution"].apply(
            lambda value: (
                "Pushes toward churn"
                if value > 0
                else "Pushes away from churn"
            )
        )


        fig_shap = px.bar(
            chart_df,
            x="SHAP contribution",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={
                "Pushes toward churn": "#FB7185",
                "Pushes away from churn": "#2DD4BF",
            },
            title="Strongest SHAP contributions",
            labels={
                "SHAP contribution": "Contribution to model output",
                "Feature": "Customer feature",
                "Direction": "Contribution direction",
            },
        )

        fig_shap.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=30),
            legend_title_text="Contribution direction",
        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True,
        )

    st.caption(
        "SHAP values explain how features influence this individual "
        "prediction. Positive values push the model output toward "
        "churn; negative values push it away. They are not "
        "percentage-point changes in churn probability."
    )

# RETENTION ACTION GUIDE

st.divider()

st.subheader("Retention Action Guide")

st.write(
    "Use customer risk and retention priority together "
    "to guide retention activities."
)

with st.expander("View recommended retention strategies"):

    st.markdown(
        """
        **High Risk + High Priority**

        Prioritise personalised outreach. Review the customer's
        purchase history and consider a relevant retention incentive.

        **High Risk + Medium Priority**

        Use targeted re-engagement campaigns to encourage another
        purchase.

        **High Risk + Low Priority**

        Start with low-cost automated communication rather than
        expensive individual outreach.

        **Medium Risk Customers**

        Focus on strengthening customer engagement through
        relevant offers, loyalty benefits, or product recommendations.

        **Low Risk Customers**

        Focus on maintaining engagement and encouraging repeat
        purchases. Consider loyalty rewards for high-priority customers.
        """
    )

    st.info(
        "These are suggested strategies, not guaranteed interventions. "
        "Evaluate campaign results before deciding which actions "
        "are most effective."
    )


# Downloads
st.divider()
st.subheader("Export customer reports")
download_left, download_right = st.columns(2)
with download_left:
    st.download_button(
        "Download filtered retention report (CSV)",
        data=priority_table.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_retention_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
with download_right:
    st.download_button(
        "Download risk summary (CSV)",
        data=spending_summary.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_risk_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "RetailPulse | Customer Churn Prediction & Retention Intelligence "
    "| Built with Streamlit, Pandas, Plotly, and machine learning."
)
