from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RetailPulse | Inventory Optimization", page_icon="📦", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
PATHS = {
    "priority": PROCESSED / "inventory_priority_dataset_normalized.csv",
    "profiles": PROCESSED / "product_demand_profiles_normalized.parquet",
    "trends": PROCESSED / "product_demand_trends_normalized.csv",
    "daily": PROCESSED / "product_daily_demand_normalized.parquet",
    "sales": PROCESSED / "clean_sales.parquet",
    "priority_summary": OUTPUTS / "inventory_review_priority_summary_normalized.csv",
    "trend_summary": OUTPUTS / "inventory_trend_summary_normalized.csv",
    "readiness": OUTPUTS / "inventory_data_readiness.csv",
}

st.markdown("""
<style>
.stApp {background:#0b1120}
[data-testid="stSidebar"] {background:#151f32}
[data-testid="stMetric"] {background:#111d31;border:1px solid #2a3b56;padding:16px 18px;border-radius:14px}
[data-testid="stMetricLabel"] {color:#a8c4ec}
[data-testid="stMetricValue"] {color:#f8fafc}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading inventory datasets...")
def read_dataset(path_string, kind):
    path = Path(path_string)
    return pd.read_parquet(path) if kind == "parquet" else pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner="Loading original sales transactions...")
def load_sales_transactions():
    columns = ["invoice", "stock_code", "description", "quantity", "invoice_date", "unit_price", "revenue", "is_cancelled"]
    sales = pd.read_parquet(PATHS["sales"], columns=columns)
    sales["stock_code"] = sales["stock_code"].astype(str).str.strip()
    sales["invoice_date"] = pd.to_datetime(sales["invoice_date"], errors="coerce")
    for col in ["quantity", "unit_price", "revenue"]:
        sales[col] = pd.to_numeric(sales[col], errors="coerce")
    sales["is_cancelled"] = sales["is_cancelled"].fillna(True).astype(bool)
    return sales


def need_files():
    missing = [str(p) for p in PATHS.values() if not p.exists()]
    if missing:
        st.error("Required inventory files are missing:\n\n" + "\n".join(f"- `{p}`" for p in missing))
        st.info("Check that processed datasets are under `data/processed/` and summary files are under `outputs/`.")
        st.stop()


def money(x):
    try:
        return f"£{float(x):,.2f}"
    except (TypeError, ValueError):
        return "—"


def number(x, decimals=0):
    try:
        return f"{float(x):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def bool_value(value):
    return False if pd.isna(value) else bool(value)


need_files()
priority = read_dataset(str(PATHS["priority"]), "csv")
profiles = read_dataset(str(PATHS["profiles"]), "parquet")
trends = read_dataset(str(PATHS["trends"]), "csv")
daily = read_dataset(str(PATHS["daily"]), "parquet")
priority_summary = read_dataset(str(PATHS["priority_summary"]), "csv")
trend_summary = read_dataset(str(PATHS["trend_summary"]), "csv")
readiness = read_dataset(str(PATHS["readiness"]), "csv")

for frame in [priority, profiles, trends, daily]:
    if "stock_code" in frame:
        frame["stock_code"] = frame["stock_code"].astype(str).str.strip()

for col in ["daily_units", "daily_revenue"]:
    daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0)
daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
daily = daily.dropna(subset=["date", "stock_code"])
daily = daily.groupby(["stock_code", "date"], as_index=False).agg(
    daily_units=("daily_units", "sum"), daily_revenue=("daily_revenue", "sum")
)

for frame in [priority, profiles, trends]:
    for col in ["units_sold", "revenue", "mean_daily_demand", "std_daily_demand", "cv_demand",
                "recent_units", "previous_units", "recent_revenue", "previous_revenue",
                "demand_change_pct", "yoy_change_pct", "active_selling_days"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

if "description" not in priority.columns:
    priority = priority.merge(profiles[["stock_code", "description"]].drop_duplicates("stock_code"), on="stock_code", how="left")
for col in ["movement_class", "trend_status", "review_priority"]:
    if col in priority:
        priority[col] = priority[col].fillna("Unknown").astype(str)
priority["description"] = priority["description"].fillna("").astype(str)

units_col = "units_sold" if "units_sold" in priority.columns else "total_historical_units"
revenue_col = "revenue" if "revenue" in priority.columns else "total_historical_revenue"
priority[units_col] = pd.to_numeric(priority[units_col], errors="coerce").fillna(0)
priority[revenue_col] = pd.to_numeric(priority[revenue_col], errors="coerce").fillna(0)

st.sidebar.markdown("## 📦 RetailPulse")
st.sidebar.caption("Inventory intelligence")
page = st.sidebar.radio("Navigate", ["Overview", "Demand Analytics", "Product Explorer", "Data Readiness"])
st.sidebar.divider()
st.sidebar.caption(f"{priority['stock_code'].nunique():,} products loaded")
st.sidebar.caption("Source: RetailPulse processed datasets")
st.sidebar.caption("Historical inventory analytics")

st.title("📦 Inventory Optimization")
st.caption("Historical demand intelligence • Product movement • Review priorities")

n_products = priority["stock_code"].nunique()
total_units = priority[units_col].sum()
total_revenue = priority[revenue_col].sum()
no_recent = int(priority["no_recent_sales_flag"].fillna(False).astype(bool).sum()) if "no_recent_sales_flag" in priority else 0
high_cv = int(priority["high_variability_flag"].fillna(False).astype(bool).sum()) if "high_variability_flag" in priority else 0

if page == "Overview":
    st.header("Executive overview")
    cols = st.columns(4)
    cols[0].metric("Products analyzed", f"{n_products:,}")
    cols[1].metric("Historical units sold", f"{total_units:,.0f}")
    cols[2].metric("Historical revenue", f"£{total_revenue / 1_000_000:.2f}M")
    cols[3].metric("No recent recorded sales", f"{no_recent:,}")
    st.caption("Revenue and units are historical aggregates, not current inventory value.")
    a, b = st.columns(2)
    with a:
        st.subheader("Product movement distribution")
        mov = priority.groupby("movement_class")["stock_code"].nunique().reindex(["Fast-moving", "Medium-moving", "Slow-moving"]).fillna(0).reset_index(name="product_count")
        fig = px.bar(mov, x="movement_class", y="product_count", text="product_count", color="movement_class",
                     color_discrete_map={"Fast-moving":"#38bdf8", "Medium-moving":"#818cf8", "Slow-moving":"#a78bfa"})
        fig.update_layout(template="plotly_dark", showlegend=False, xaxis_title="Movement class", yaxis_title="Products")
        st.plotly_chart(fig, use_container_width=True, key="overview_movement")
    with b:
        st.subheader("Products by review priority")
        pc = priority.groupby("review_priority")["stock_code"].nunique().sort_values().reset_index(name="product_count")
        fig = px.bar(pc, x="product_count", y="review_priority", orientation="h", text="product_count", color="product_count", color_continuous_scale="Blues")
        fig.update_layout(template="plotly_dark", showlegend=False, coloraxis_showscale=False, xaxis_title="Products", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key="overview_priorities")
    st.subheader("Review priority summary")
    summary = priority_summary.copy()
    display_names = {
        "review_priority": "Review priority",
        "product_count": "Products",
        "total_units": "Historical units sold",
        "total_revenue": "Historical revenue",
        "product_share": "Share of products",
    }
    summary = summary.rename(columns={k: v for k, v in display_names.items() if k in summary.columns})
    if "Historical units sold" in summary.columns:
        summary["Historical units sold"] = pd.to_numeric(summary["Historical units sold"], errors="coerce").map(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "—"
        )
    if "Historical revenue" in summary.columns:
        summary["Historical revenue"] = pd.to_numeric(summary["Historical revenue"], errors="coerce").map(money)
    if "Share of products" in summary.columns:
        share = pd.to_numeric(summary["Share of products"], errors="coerce")
        # The saved summary stores product_share as a fraction (for example, 0.061 = 6.1%).
        summary["Share of products"] = share.map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.info("Review categories use a primary-priority hierarchy. Underlying flags can overlap and should be assessed separately.")

elif page == "Demand Analytics":
    st.header("Demand Analytics")
    st.caption("Compare recent activity with the previous period and inspect historical demand patterns.")

    metric = st.selectbox("Compare metric", ["Units sold", "Revenue (£)"], key="demand_metric")
    recent = "recent_units" if metric == "Units sold" else "recent_revenue"
    previous = "previous_units" if metric == "Units sold" else "previous_revenue"

    if {recent, previous}.issubset(trends.columns):
        recent_total = pd.to_numeric(trends[recent], errors="coerce").sum()
        previous_total = pd.to_numeric(trends[previous], errors="coerce").sum()
        change = ((recent_total / previous_total) - 1) * 100 if previous_total else float("nan")
        k = st.columns(3)
        if metric == "Units sold":
            k[0].metric("Recent 90-day units", number(recent_total))
            k[1].metric("Previous 90-day units", number(previous_total))
        else:
            k[0].metric("Recent 90-day revenue", money(recent_total))
            k[1].metric("Previous 90-day revenue", money(previous_total))
        k[2].metric("Change vs previous period", f"{change:+.1f}%" if pd.notna(change) else "Not available")

    if {"movement_class", recent, previous}.issubset(trend_summary.columns):
        long = trend_summary.melt(id_vars="movement_class", value_vars=[recent, previous], var_name="period", value_name="value")
        long["period"] = long["period"].map({recent: "Recent 90 days", previous: "Previous 90 days"})
        fig = px.bar(long, x="movement_class", y="value", color="period", barmode="group",
                     color_discrete_map={"Recent 90 days": "#38bdf8", "Previous 90 days": "#818cf8"},
                     labels={"movement_class": "Movement class", "value": metric})
        if metric == "Revenue (£)":
            fig.update_yaxes(tickprefix="£", tickformat="~s")
        else:
            fig.update_yaxes(tickformat="~s")
        fig.update_layout(template="plotly_dark", legend_title_text="Period", xaxis_title="Movement class")
        st.plotly_chart(fig, use_container_width=True, key="recent_previous_demand")
    else:
        st.warning("The trend summary does not contain the columns needed for this comparison.")

    st.subheader("Year-over-year demand change")
    if "yoy_trend_status" in trends:
        yoy = trends["yoy_trend_status"].fillna("Insufficient comparable baseline").value_counts().rename_axis("trend").reset_index(name="products")
        fig = px.pie(yoy, names="trend", values="products", hole=0.55, color="trend",
                     color_discrete_map={"Declining vs same season": "#fb7185", "Increasing vs same season": "#34d399",
                                         "Relatively stable vs same season": "#94a3b8", "Insufficient comparable baseline": "#64748b",
                                         "No recent recorded demand": "#fbbf24"})
        fig.update_traces(textposition="inside", texttemplate="%{percent:.1%}", hovertemplate="%{label}<br>Products: %{value:,}<br>Share: %{percent:.1%}<extra></extra>")
        fig.update_layout(template="plotly_dark", legend_title_text="Year-over-year status")
        st.plotly_chart(fig, use_container_width=True, key="yoy_distribution")
        st.caption("Year-over-year labels compare recorded sales with the same seasonal window. Insufficient baselines and no recent recorded demand are not growth or decline signals.")

    st.subheader("Demand variability")
    if {"mean_daily_demand", "cv_demand", "movement_class"}.issubset(priority.columns):
        scatter = priority.copy()
        scatter = scatter[scatter["mean_daily_demand"].gt(0) & scatter["cv_demand"].notna()]
        if not scatter.empty:
            fig = px.scatter(scatter, x="mean_daily_demand", y="cv_demand", color="movement_class", log_x=True,
                             hover_name="description", hover_data={"stock_code": True, units_col: ":,", revenue_col: ":,.2f", "mean_daily_demand": ":,.3f", "cv_demand": ":,.3f"}, opacity=0.72,
                             labels={"mean_daily_demand": "Mean daily demand (log scale)", "cv_demand": "Demand coefficient of variation (CV)", "movement_class": "Movement class"})
            fig.update_layout(template="plotly_dark", legend_title_text="Movement class")
            st.plotly_chart(fig, use_container_width=True, key="demand_variability")
    st.caption("High variability can reflect intermittent demand or bulk orders. Treat these as review signals, not direct reorder recommendations.")

elif page == "Product Explorer":
    st.header("Product Explorer")
    st.caption("Filter historical products and investigate a selected product.")
    c1, c2, c3 = st.columns(3)
    movements = sorted(priority["movement_class"].dropna().unique().tolist())
    trends_list = sorted(priority["trend_status"].dropna().unique().tolist()) if "trend_status" in priority else []
    priorities = sorted(priority["review_priority"].dropna().unique().tolist())
    with c1:
        sel_mov = st.multiselect("Movement class", movements, default=movements, key="filter_movement")
    with c2:
        sel_trend = st.multiselect("Demand trend", trends_list, default=trends_list, key="filter_trend") if trends_list else []
    with c3:
        sel_priority = st.multiselect("Review priority", priorities, default=priorities, key="filter_priority")
    query = st.text_input("Search stock code or description", key="product_search")
    sort_map = {"Historical revenue": revenue_col, "Historical units": units_col, "Mean daily demand": "mean_daily_demand", "Demand variability (CV)": "cv_demand"}
    sort_label = st.selectbox("Sort products by", list(sort_map), key="sort_products")
    filtered = priority[priority["movement_class"].isin(sel_mov) & priority["review_priority"].isin(sel_priority)].copy()
    if "trend_status" in filtered:
        filtered = filtered[filtered["trend_status"].isin(sel_trend)]
    if query.strip():
        q = query.strip().lower()
        filtered = filtered[filtered["stock_code"].str.lower().str.contains(q, na=False) | filtered["description"].str.lower().str.contains(q, na=False)]
    sort_col = sort_map[sort_label]
    if sort_col not in filtered:
        filtered[sort_col] = pd.NA
    filtered = filtered.sort_values(sort_col, ascending=False, na_position="last")
    st.caption(f"{len(filtered):,} products match your filters.")
    display_columns = {
        "stock_code": "Stock code",
        "description": "Product description",
        "movement_class": "Movement",
        units_col: "Historical units",
        revenue_col: "Historical revenue",
        "trend_status": "Demand trend",
        "review_priority": "Review priority",
    }
    show_cols = [c for c in ["stock_code", "description", "movement_class", units_col, revenue_col, "trend_status", "review_priority"] if c in filtered]
    table = filtered[show_cols].rename(columns=display_columns).copy()
    table_config = {
        "Stock code": st.column_config.TextColumn("Stock code", width="small"),
        "Product description": st.column_config.TextColumn("Product description", width="large"),
        "Movement": st.column_config.TextColumn("Movement", width="small"),
        "Historical units": st.column_config.NumberColumn("Historical units", format="%,.0f", width="small"),
        "Historical revenue": st.column_config.NumberColumn("Historical revenue (£)", format="£%,.2f", width="medium"),
        "Demand trend": st.column_config.TextColumn("Demand trend", width="medium"),
        "Review priority": st.column_config.TextColumn("Review priority", width="medium"),
    }
    st.dataframe(table, use_container_width=True, hide_index=True, height=350, column_config=table_config)
    st.download_button("Download filtered products as CSV", filtered.to_csv(index=False).encode("utf-8"), "retailpulse_filtered_products.csv", "text/csv", key="download_filtered")

    st.divider()
    st.subheader("Product detail")
    options = priority.copy()
    options["selector"] = options["stock_code"].astype(str) + " | " + options["description"].fillna("").astype(str)
    labels = options["selector"].tolist()
    default_idx = 0
    if query.strip():
        match_idx = options[options["stock_code"].str.contains(query.strip(), case=False, na=False) | options["description"].str.contains(query.strip(), case=False, na=False)].index
        if len(match_idx):
            default_idx = options.index.get_loc(match_idx[0])
    selected_label = st.selectbox("Select a product", labels, index=min(default_idx, len(labels)-1), key="detail_product")
    selected = options.loc[options["selector"].eq(selected_label)].iloc[0]
    code = str(selected["stock_code"]).strip()
    desc = selected["description"] or code
    st.markdown(f"### {desc}")
    st.caption(f"Stock code: {code}")
    m = st.columns(3)
    m[0].metric("Historical units", number(selected.get(units_col, 0)))
    m[1].metric("Historical revenue", money(selected.get(revenue_col, 0)))
    m[2].metric("Mean daily demand", number(selected.get("mean_daily_demand", 0), 3))
    m = st.columns(3)
    m[0].markdown("**Movement class**")
    m[0].write(str(selected.get("movement_class", "Unknown")))
    m[1].markdown("**Demand trend**")
    m[1].write(str(selected.get("trend_status", "Unknown")))
    m[2].markdown("**Review priority**")
    m[2].write(str(selected.get("review_priority", "Unknown")))
    if bool_value(selected.get("limited_history_flag", False)):
        st.info("Limited observed history: interpret the demand average and variability cautiously.")
    flags = [("fast_moving_flag", "Fast-moving"), ("limited_history_flag", "Limited observed history"), ("no_recent_sales_flag", "No recent recorded sales"), ("growth_signal_flag", "Growth signal"), ("sustained_decline_flag", "Sustained historical decline"), ("high_variability_flag", "High demand variability")]
    signals = [label for col, label in flags if col in selected.index and bool_value(selected[col])]
    st.markdown("**Review signals**")
    if signals:
        for signal in signals:
            st.write(f"• {signal}")
    else:
        st.caption("No independent review flags are set for this product.")
    st.caption("Historical signals are for investigation, not proof of a stockout or a direct replenishment recommendation.")

    product_daily = daily[daily["stock_code"].eq(code)].copy().sort_values("date")
    if product_daily.empty:
        st.info("No product-level daily demand records were found for this product.")
    else:
        product_daily = product_daily.groupby("date", as_index=False).agg(daily_units=("daily_units", "sum"), daily_revenue=("daily_revenue", "sum"))
        first_date, last_date = product_daily["date"].min().date(), product_daily["date"].max().date()
        active_full = int(product_daily["daily_units"].gt(0).sum())
        single_day = active_full == 1
        st.divider()
        st.subheader("Historical demand over time")
        if single_day:
            st.warning("Single-day sales concentration: all recorded units occurred on one sales day. Repeated demand history is insufficient to estimate normal daily demand.")
            st.metric("Single-day recorded units", number(product_daily["daily_units"].sum()))
        date_range = st.date_input("Select date range", value=(first_date, last_date), min_value=first_date, max_value=last_date, key=f"date_range_v2_{code}")
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
        else:
            start = end = date_range
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        period = product_daily[product_daily["date"].between(start_ts, end_ts)].copy()
        if period.empty:
            st.info("No product records fall within the selected date range.")
        else:
            full_dates = pd.date_range(start_ts, end_ts, freq="D")
            period = period.set_index("date").reindex(full_dates, fill_value=0).rename_axis("date").reset_index()
            for col in ["daily_units", "daily_revenue"]:
                period[col] = pd.to_numeric(period[col], errors="coerce").fillna(0)
            k = st.columns(3)
            k[0].metric("Days in selected period", f"{len(period):,}")
            k[1].metric("Recorded sales days", f"{int(period['daily_units'].gt(0).sum()):,}")
            k[2].metric("Median daily units", number(period["daily_units"].median()))
            metric = st.radio("Chart metric", ["Daily units", "Daily revenue"], horizontal=True, key=f"chart_metric_{code}")
            ycol = "daily_units" if metric == "Daily units" else "daily_revenue"
            ylabel = "Units sold" if metric == "Daily units" else "Revenue (£)"
            if single_day:
                event = product_daily.loc[product_daily["daily_units"].gt(0)].iloc[0]
                st.info(f"Recorded sales event: {event['date']:%d %b %Y}. {event['daily_units']:,.0f} units were recorded on this date. A trend line and rolling average are not meaningful with only one active sales day.")
            else:
                period["rolling_7_day"] = period[ycol].rolling(7, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=period["date"], y=period[ycol], mode="lines", name=ylabel, line=dict(color="#38bdf8", width=1.5), hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=period["date"], y=period["rolling_7_day"], mode="lines", name="7-day average", line=dict(color="#fbbf24", width=2), hovertemplate="%{x|%d %b %Y}<br>7-day average: %{y:,.2f}<extra></extra>"))
                fig.update_layout(template="plotly_dark", title=f"Demand history — {code}", xaxis_title="Date", yaxis_title=ylabel, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True, key=f"history_{code}_{metric}")
            k = st.columns(4)
            k[0].metric("Recorded units", number(product_daily["daily_units"].sum()))
            k[1].metric("Days with sales", f"{active_full:,}")
            k[2].metric("Zero-sales days", f"{max((last_date-first_date).days + 1-active_full, 0):,}")
            k[3].metric("Peak daily units", number(product_daily["daily_units"].max()))
            st.subheader("Largest recorded sales days")
            largest = product_daily[product_daily["daily_units"].gt(0)].nlargest(10, "daily_units").copy()
            largest = largest.rename(columns={"date": "Date", "daily_units": "Units sold", "daily_revenue": "Revenue (£)"})
            st.dataframe(largest, use_container_width=True, hide_index=True, column_config={
                "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "Units sold": st.column_config.NumberColumn("Units sold", format="%,.0f"),
                "Revenue (£)": st.column_config.NumberColumn("Revenue (£)", format="£%,.2f"),
            })
            st.download_button("Download this product's daily history", product_daily.to_csv(index=False).encode("utf-8"), f"product_{code}_daily_history.csv", "text/csv", key=f"download_daily_{code}")
            st.caption("Zero-sales days are filled only between the product's first and last recorded sales dates. Dates outside this observed period are not assumed to represent demand.")

            st.subheader("Demand anomaly investigation")
            anomaly_cols = ["date", "daily_units", "daily_revenue", "baseline_median", "anomaly_threshold", "anomaly_reason"]
            anomalies = pd.DataFrame(columns=anomaly_cols)
            positive = product_daily.loc[product_daily["daily_units"].gt(0), "daily_units"]
            if active_full >= 5:
                med = float(positive.median())
                mad = float((positive - med).abs().median())
                threshold = max(med + 3 * 1.4826 * mad, med * 3)
                mask = product_daily["daily_units"].gt(threshold) & product_daily["date"].between(start_ts, end_ts)
                anomalies = product_daily.loc[mask, ["date", "daily_units", "daily_revenue"]].copy()
                if not anomalies.empty:
                    anomalies["baseline_median"] = med
                    anomalies["anomaly_threshold"] = threshold
                    anomalies["anomaly_reason"] = "Above robust median/MAD review threshold"
            elif single_day:
                st.info("No statistical anomalies can be established from one active sales day. The recorded event can still be inspected below.")
            else:
                st.info("At least five active sales days are required for this anomaly screening.")
            if not anomalies.empty:
                st.warning(f"{len(anomalies):,} sales day(s) flagged for review. These are statistical signals, not confirmed errors.")
                st.dataframe(anomalies, use_container_width=True, hide_index=True, column_config={"date":st.column_config.DateColumn("Date"), "daily_units":st.column_config.NumberColumn("Units sold", format="%d"), "daily_revenue":st.column_config.NumberColumn("Revenue (£)", format="£%.2f"), "baseline_median":st.column_config.NumberColumn("Historical median units", format="%.2f"), "anomaly_threshold":st.column_config.NumberColumn("Flag threshold", format="%.2f"), "anomaly_reason":"Reason for review"})
            elif active_full >= 5:
                st.success("No unusually high sales days were flagged in the selected period.")
            st.download_button("Download anomaly investigation CSV", anomalies.to_csv(index=False).encode("utf-8"), f"product_{code}_anomaly_review.csv", "text/csv", key=f"download_anomalies_{code}")

            st.divider()
            st.subheader("Original transaction investigation")
            if single_day:
                dates_to_investigate = product_daily.loc[product_daily["daily_units"].gt(0), "date"].dt.strftime("%Y-%m-%d").tolist()
            elif not anomalies.empty:
                dates_to_investigate = anomalies["date"].dt.strftime("%Y-%m-%d").tolist()
            else:
                dates_to_investigate = period.loc[period["daily_units"].gt(0), "date"].dt.strftime("%Y-%m-%d").tolist()
            if dates_to_investigate:
                chosen_date = st.selectbox("Select a recorded sales date", dates_to_investigate, key=f"transaction_date_{code}")
                if st.button("Load original transactions", key=f"load_transactions_{code}"):
                    st.session_state[f"transactions_loaded_{code}"] = chosen_date
                if st.session_state.get(f"transactions_loaded_{code}") == chosen_date:
                    sales = load_sales_transactions()
                    tx = sales.loc[sales["stock_code"].eq(code) & sales["invoice_date"].dt.normalize().eq(pd.Timestamp(chosen_date)) & sales["quantity"].gt(0) & sales["unit_price"].gt(0) & ~sales["is_cancelled"]].copy()
                    if tx.empty:
                        st.warning("No qualifying original sales transactions were found for this product and date.")
                    else:
                        tx["calculated_revenue"] = tx["quantity"] * tx["unit_price"]
                        total_tx_units = float(tx["quantity"].sum())
                        total_tx_revenue = float(tx["calculated_revenue"].sum())
                        invoice_count = tx["invoice"].nunique()
                        largest_invoice_units = float(tx.groupby("invoice")["quantity"].sum().max())
                        share = largest_invoice_units / total_tx_units * 100 if total_tx_units else 0
                        k = st.columns(3)
                        k[0].metric("Matching transactions", f"{len(tx):,}")
                        k[1].metric("Distinct invoices", f"{invoice_count:,}")
                        k[2].metric("Total units", number(total_tx_units))
                        st.metric("Calculated transaction revenue", money(total_tx_revenue))
                        st.metric("Largest invoice share of daily units", f"{share:.2f}%")
                        tx_cols = ["invoice_date", "invoice", "stock_code", "description", "quantity", "unit_price", "calculated_revenue"]
                        st.dataframe(tx[tx_cols].sort_values("quantity", ascending=False), use_container_width=True, hide_index=True, column_config={"invoice_date":st.column_config.DatetimeColumn("Invoice date and time"), "quantity":st.column_config.NumberColumn("Quantity", format="%d"), "unit_price":st.column_config.NumberColumn("Unit price (£)", format="£%.2f"), "calculated_revenue":st.column_config.NumberColumn("Calculated revenue (£)", format="£%.2f")})
                        st.download_button("Download original transactions", tx[tx_cols].to_csv(index=False).encode("utf-8"), f"product_{code}_{chosen_date}_transactions.csv", "text/csv", key=f"download_tx_{code}")
                        st.caption("Invoice concentration is descriptive and does not establish why the order occurred.")
            else:
                st.info("No recorded sales dates are available for transaction investigation.")

elif page == "Data Readiness":
    st.header("Inventory data readiness")
    st.caption("Capabilities supported by current files and the gaps that remain.")
    counts = readiness["status"].fillna("Unknown").value_counts() if "status" in readiness else pd.Series(dtype=int)
    a, b, c = st.columns(3)
    a.metric("Available", f"{int(counts.get('Available', 0)):,}")
    b.metric("Unavailable", f"{int(counts.get('Unavailable', 0)):,}")
    c.metric("Blocked by missing inputs", f"{int(counts.get('Blocked by missing inputs', 0)):,}")
    st.subheader("Capability status and required inputs")
    core_columns = [c for c in ["capability", "status", "required_data"] if c in readiness]
    core = readiness[core_columns].rename(columns={"capability": "Capability", "status": "Status", "required_data": "Required data"})
    st.dataframe(core, use_container_width=True, hide_index=True, height=300, column_config={
        "Capability": st.column_config.TextColumn("Capability", width="medium"),
        "Status": st.column_config.TextColumn("Status", width="small"),
        "Required data": st.column_config.TextColumn("Required data", width="large"),
    })
    detail_columns = [c for c in ["capability", "current_evidence", "limitation"] if c in readiness]
    details = readiness[detail_columns].rename(columns={"capability": "Capability", "current_evidence": "Evidence available", "limitation": "Current limitation"})
    st.subheader("Evidence and limitations")
    st.dataframe(details, use_container_width=True, hide_index=True, height=300, column_config={
        "Capability": st.column_config.TextColumn("Capability", width="medium"),
        "Evidence available": st.column_config.TextColumn("Evidence available", width="large"),
        "Current limitation": st.column_config.TextColumn("Current limitation", width="large"),
    })
    st.warning("Current stock balances, supplier lead times, purchase-order data, and confirmed stockout records are unavailable. Safety stock, reorder points, and order quantities cannot be calculated reliably from these files alone.")
    st.download_button("Download data readiness report", readiness.to_csv(index=False).encode("utf-8"), "inventory_data_readiness.csv", "text/csv", key="download_readiness")

st.divider()
st.caption("RetailPulse • Historical inventory analytics. Historical sales are not current stock on hand. Zero sales alone do not prove a stockout.")
