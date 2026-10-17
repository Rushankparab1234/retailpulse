
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st



# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
OUTPUT_DIR = BASE_DIR / "outputs"

FORECAST_FILE = DATA_DIR / "final_demand_forecast.csv"
HISTORY_FILE = DATA_DIR / "daily_demand.parquet"
MODEL_COMPARISON_FILE = DATA_DIR / "final_model_comparison.csv"
METRICS_FILE = NOTEBOOK_DIR / "final_forecasting_metrics.json"
CV_FILE = OUTPUT_DIR / "demand_cv_fold_metrics.csv"

BG = "#0B1120"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"


# RetailPulse chart palette
CHART_COLORS = [
    "#2DD4BF",  # Teal
    "#FBBF24",  # Amber
    "#A78BFA",  # Violet
    "#FB7185",  # Coral
    "#34D399",  # Green
    "#FB923C",  # Orange
    "#60A5FA",  # Sky blue
]

MODEL_COLORS = {
    "Prophet Smooth": "#FBBF24",
    "Prophet + LSTM Hybrid": "#A78BFA",
}

ERROR_COLORS = {
    "Prophet Error": "#FB923C",
    "Hybrid Error": "#2DD4BF",
}



st.title("Demand Forecasting")
st.caption(
    "RetailPulse | Historical demand, forecast performance, "
    "model comparison, and time-series validation"
)


# DATA LOADING


@st.cache_data(show_spinner="Loading forecasting artifacts...")
def load_forecast(path: str, modified: float) -> pd.DataFrame:
    del modified
    df = pd.read_csv(path, parse_dates=["date"])

    required = [
        "date",
        "actual_demand",
        "prophet_forecast",
        "hybrid_forecast",
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing forecast columns: {missing}")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate forecast dates detected.")

    numeric_cols = [
        "actual_demand",
        "prophet_forecast",
        "hybrid_forecast",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[required].isna().any().any():
        raise ValueError("Missing or invalid values found in required forecast columns.")

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner="Loading historical demand...")
def load_history(path: str, modified: float) -> pd.DataFrame:
    del modified
    df = pd.read_parquet(path)

    required = ["ds", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing historical demand columns: {missing}")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")

    return df


@st.cache_data(show_spinner="Loading model metrics...")
def load_json(path: str, modified: float) -> dict:
    del modified
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading comparison results...")
def load_csv(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path)


# Validate required artifacts before rendering.
required_files = [FORECAST_FILE, HISTORY_FILE, METRICS_FILE]

missing_files = [str(p) for p in required_files if not p.exists()]

if missing_files:
    st.error("Required forecasting artifacts are missing:")
    for filename in missing_files:
        st.write(f"- `{filename}`")
    st.stop()

try:
    forecast = load_forecast(str(FORECAST_FILE), FORECAST_FILE.stat().st_mtime)
    history = load_history(str(HISTORY_FILE), HISTORY_FILE.stat().st_mtime)
    metrics = load_json(str(METRICS_FILE), METRICS_FILE.stat().st_mtime)
except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
    st.error(f"Could not load forecasting artifacts: {exc}")
    st.stop()


# SIDEBAR FILTERS


st.sidebar.title("Demand Forecasting")
st.sidebar.caption("Historical demand and model evaluation")
st.sidebar.divider()

forecast_start = forecast["date"].min().date()
forecast_end = forecast["date"].max().date()

selected_dates = st.sidebar.date_input(
    "Forecast evaluation period",
    value=(forecast_start, forecast_end),
    min_value=forecast_start,
    max_value=forecast_end,
)

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
    start_date = end_date = selected_dates[0]
else:
    start_date = end_date = selected_dates

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

filtered = forecast[
    forecast["date"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No forecast observations exist in this date range.")
    st.stop()



# OVERVIEW METRICS


selected_model = metrics.get("selected_model", "Not specified")

st.subheader("Forecast Overview")

st.caption(
    f"Saved selected model: {selected_model} | "
    f"Saved evaluation period: {metrics.get('test_start', 'N/A')} "
    f"to {metrics.get('test_end', 'N/A')}"
)

total_actual = filtered["actual_demand"].sum()
mean_actual = filtered["actual_demand"].mean()
zero_days = int((filtered["actual_demand"] == 0).sum())

m1, m2, m3, m4 = st.columns(4)

m1.metric("Evaluation Days", f"{len(filtered):,}")
m2.metric("Total Actual Demand", f"{total_actual:,.0f}")
m3.metric("Average Daily Demand", f"{mean_actual:,.0f}")
m4.metric("Zero-Demand Days", f"{zero_days:,}")

st.divider()



# ACTUAL VS FORECAST


st.subheader("Actual Demand vs Forecast")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered["date"],
        y=filtered["actual_demand"],
        mode="lines",
        name="Actual Demand",
        line=dict(color="#2DD4BF", width=2),
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered["date"],
        y=filtered["prophet_forecast"],
        mode="lines",
        name="Prophet Smooth",
        line=dict(color="#FBBF24", width=1.5),
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered["date"],
        y=filtered["hybrid_forecast"],
        mode="lines",
        name="Prophet + LSTM Hybrid",
        line=dict(color="#A78BFA", width=1.7),
    )
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font_color=TEXT,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Daily Demand",
    legend_title="Series",
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Forecast values are displayed as saved, including any negative "
    "Prophet predictions. No clipping or recalculation is applied."
)



# VERIFIED FINAL TEST METRICS


st.divider()
st.subheader("Final Test-Period Model Performance")

st.caption(
    "These metrics describe the complete saved test period, "
    "not only the date range selected in the sidebar."
)

hybrid_cols = st.columns(4)

hybrid_metrics = [
    ("MAE", metrics.get("MAE")),
    ("RMSE", metrics.get("RMSE")),
    ("MAPE (nonzero actuals)", metrics.get("MAPE")),
    ("WAPE", metrics.get("WAPE")),
]

for container, (label, value) in zip(hybrid_cols, hybrid_metrics):
    with container:
        if value is None:
            st.metric(label, "N/A")
        elif label in ["MAPE (nonzero actuals)", "WAPE"]:
            st.metric(label, f"{float(value):.2f}%")
        else:
            st.metric(label, f"{float(value):,.2f}")

st.caption(
    "The saved MAPE excludes observations with zero actual demand. "
    "WAPE is calculated as total absolute error divided by total "
    "absolute actual demand."
)



# MODEL COMPARISON


st.divider()
st.subheader("Model Comparison")

if MODEL_COMPARISON_FILE.exists():
    try:
        model_comparison = load_csv(
            str(MODEL_COMPARISON_FILE),
            MODEL_COMPARISON_FILE.stat().st_mtime,
        )

        expected = ["Model", "MAE", "RMSE", "MAPE", "WAPE"]
        missing = [c for c in expected if c not in model_comparison.columns]

        if missing:
            st.warning(f"Model comparison columns missing: {missing}")
        else:
            st.dataframe(
                model_comparison.round(3),
                use_container_width=True,
                hide_index=True,
            )

            metric_choice = st.selectbox(
                "Compare models by metric",
                ["MAE", "RMSE", "MAPE", "WAPE"],
                key="forecast_model_metric",
            )


            chart = px.bar(
                model_comparison,
                x="Model",
                y=metric_choice,
                color="Model",
                color_discrete_map=MODEL_COLORS,
                text=metric_choice,
                title=f"Model Comparison — {metric_choice}",
                template="plotly_dark",
            )


            chart.update_layout(
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                font_color=TEXT,
                showlegend=False,
            )

            chart.update_traces(texttemplate="%{text:.2f}", textposition="outside")

            st.plotly_chart(chart, use_container_width=True)

            st.caption(
                "Lower values indicate smaller errors for all four metrics. "
                "MAPE values are shown as percentages."
            )

    except (ValueError, OSError, KeyError) as exc:
        st.warning(f"Could not load model comparison: {exc}")
else:
    st.info("Model comparison CSV not found; skipping this section.")


# CROSS-VALIDATION


st.divider()
st.subheader("Expanding-Window Cross-Validation")

if CV_FILE.exists():
    try:
        cv = load_csv(str(CV_FILE), CV_FILE.stat().st_mtime)

        expected_cv = ["Fold", "MAE", "RMSE", "MAPE", "WAPE"]
        missing = [c for c in expected_cv if c not in cv.columns]

        if missing:
            st.warning(f"Cross-validation columns missing: {missing}")
        else:
            cv = cv[expected_cv].copy()

            for col in expected_cv:
                cv[col] = pd.to_numeric(cv[col], errors="coerce")

            if cv[expected_cv].isna().any().any():
                st.warning(
                    "The cross-validation file contains missing or "
                    "non-numeric values."
                )
            else:
                st.dataframe(
                    cv.round(3),
                    use_container_width=True,
                    hide_index=True,
                )

                cv_metric = st.selectbox(
                    "Cross-validation metric",
                    ["MAE", "RMSE", "MAPE", "WAPE"],
                    key="cv_metric",
                )


                cv_fig = px.bar(
                    cv,
                    x="Fold",
                    y=cv_metric,
                    color="Fold",
                    color_discrete_sequence=CHART_COLORS,
                    text=cv_metric,
                    title=f"{cv_metric} by Validation Fold",
                    template="plotly_dark",
                )

                cv_fig.update_layout(
                    paper_bgcolor=BG,
                    plot_bgcolor=BG,
                    font_color=TEXT,
                    xaxis_title="Validation Fold",
                    yaxis_title=cv_metric,
                    showlegend=False,
                )

                cv_fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside",
                )


                st.plotly_chart(cv_fig, use_container_width=True)

                st.caption(
                    "Fold results are the saved reported metrics from the "
                    "notebook. Some values in the source comparison table "
                    "were manually entered and have not been independently "
                    "recomputed here."
                )

    except (ValueError, OSError, KeyError) as exc:
        st.warning(f"Could not load cross-validation results: {exc}")
else:
    st.info(
        "Cross-validation file not found. Expected: "
        "outputs/demand_cv_fold_metrics.csv"
    )



# RESIDUAL AND ERROR ANALYSIS


st.divider()
st.subheader("Forecast Error Analysis")

error_df = filtered.copy()

error_df["Prophet Error"] = (error_df["actual_demand"] - error_df["prophet_forecast"])
error_df["Hybrid Error"] = (error_df["actual_demand"] - error_df["hybrid_forecast"])
error_df["Hybrid Absolute Error"] = error_df["Hybrid Error"].abs()

e1, e2 = st.columns(2)

with e1:
    prophet_error_fig = px.histogram(
        error_df,
        x="Prophet Error",
        nbins=30,
        color_discrete_sequence=["#FB923C"],
        title="Prophet Forecast Error Distribution",
        template="plotly_dark",
    )
    prophet_error_fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color=TEXT,
    )
    st.plotly_chart(prophet_error_fig, use_container_width=True)

with e2:
    hybrid_error_fig = px.histogram(
        error_df,
        x="Hybrid Error",
        nbins=30,
        color_discrete_sequence=["#2DD4BF"],
        title="Hybrid Forecast Error Distribution",
        template="plotly_dark",
    )

    hybrid_error_fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color=TEXT,
    )
    st.plotly_chart(hybrid_error_fig, use_container_width=True)


# Weekday-level error analysis
if "day_of_week" in error_df.columns:
    st.subheader("Hybrid Error by Day of Week")

    weekday_summary = (
        error_df.groupby("day_of_week", as_index=False)
        .agg(
            Observations=("Hybrid Error", "size"),
            MAE=("Hybrid Absolute Error", "mean"),
            Mean_Error=("Hybrid Error", "mean"),
        )
    )


    weekday_colors = {
        "Monday": "#2DD4BF",
        "Tuesday": "#FBBF24",
        "Wednesday": "#A78BFA",
        "Thursday": "#FB7185",
        "Friday": "#FB923C",
        "Saturday": "#34D399",
        "Sunday": "#60A5FA",
    }

    weekday_fig = px.bar(
        weekday_summary,
        x="day_of_week",
        y="MAE",
        color="day_of_week",
        color_discrete_map=weekday_colors,
        category_orders={
            "day_of_week": [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ]
        },
        text="MAE",
        title="Mean Absolute Error by Weekday",
        template="plotly_dark",
    )

    weekday_fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color=TEXT,
        xaxis_title="Day of Week",
        yaxis_title="Mean Absolute Error",
        showlegend=False,
    )

    weekday_fig.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
    )


    st.plotly_chart(weekday_fig, use_container_width=True)
    st.dataframe(weekday_summary.round(2), use_container_width=True, hide_index=True)



# DOWNLOADS

st.divider()
st.subheader("Export Reports")

download_cols = st.columns(2)

with download_cols[0]:
    st.download_button(
        "Download Filtered Forecasts",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_filtered_demand_forecast.csv",
        mime="text/csv",
    )

with download_cols[1]:
    if CV_FILE.exists():
        st.download_button(
            "Download Cross-Validation Results",
            data=CV_FILE.read_bytes(),
            file_name="demand_cv_fold_metrics.csv",
            mime="text/csv",
        )

st.caption(
    "RetailPulse Demand Forecasting | Existing saved artifacts only; "
    "no model training is performed by this dashboard."
)
