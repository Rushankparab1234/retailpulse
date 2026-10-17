
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# CONFIGURATION


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

SEGMENTS_FILE = DATA_DIR / "customer_rfm_segments.parquet"
PROFILE_FILE = DATA_DIR / "customer_segment_profile.csv"

st.title("Customer Segmentation")
st.caption(
    "Explore customer behaviour using RFM analysis "
    "and business-defined customer segments."
)

st.divider()



# DATA LOADING

@st.cache_data(show_spinner="Loading customer segmentation data...")
def load_segmentation_data(
    segments_path: str,
    segments_mtime: float,
    profile_path: str,
    profile_mtime: float,
):
    # File modification times invalidate the cache when files change.
    del segments_mtime, profile_mtime

    customers = pd.read_parquet(segments_path)
    profile = pd.read_csv(profile_path)

    return customers, profile


if not SEGMENTS_FILE.exists():
    st.error(f"Segmentation file not found: {SEGMENTS_FILE}")
    st.info(
        "Check that customer_rfm_segments.parquet exists "
        "inside data/processed."
    )
    st.stop()

if not PROFILE_FILE.exists():
    st.error(f"Segment profile not found: {PROFILE_FILE}")
    st.stop()

try:
    customers, profile = load_segmentation_data(
        str(SEGMENTS_FILE),
        SEGMENTS_FILE.stat().st_mtime,
        str(PROFILE_FILE),
        PROFILE_FILE.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not load segmentation data: {exc}")
    st.stop()



# SCHEMA VALIDATION

required_customer_columns = [
    "customer_id",
    "Recency",
    "Frequency",
    "Monetary",
    "R_Score",
    "F_Score",
    "M_Score",
    "RFM_Score",
    "Segment",
]

required_profile_columns = [
    "Segment",
    "Customers",
    "Avg_Recency",
    "Avg_Frequency",
    "Avg_Monetary",
    "Avg_RFM_Score",
    "Customer_Percentage",
]

missing_customers = [
    col for col in required_customer_columns
    if col not in customers.columns
]

missing_profile = [
    col for col in required_profile_columns
    if col not in profile.columns
]

if missing_customers or missing_profile:
    if missing_customers:
        st.error(f"Missing customer columns: {missing_customers}")

    if missing_profile:
        st.error(f"Missing profile columns: {missing_profile}")

    st.stop()

if customers["customer_id"].duplicated().any():
    st.error("Duplicate customer IDs found in segmentation data.")
    st.stop()

if customers["Segment"].isna().any():
    st.error("Missing business segment labels were found.")
    st.stop()


# SIDEBAR FILTERS

st.sidebar.title("Customer Segmentation")
st.sidebar.caption("RFM customer intelligence")
st.sidebar.divider()

segment_options = sorted(customers["Segment"].unique().tolist())

selected_segments = st.sidebar.multiselect(
    "Business segments",
    options=segment_options,
    default=segment_options,
)

customer_search = st.sidebar.text_input(
    "Search customer ID",
    placeholder="Enter customer ID",
)

min_recency = int(customers["Recency"].min())
max_recency = int(customers["Recency"].max())

recency_range = st.sidebar.slider(
    "Recency range (days)",
    min_value=min_recency,
    max_value=max_recency,
    value=(min_recency, max_recency),
)


# FILTER DATA

filtered = customers[
    customers["Segment"].isin(selected_segments)
    & customers["Recency"].between(
        recency_range[0],
        recency_range[1],
    )
].copy()

if customer_search.strip():
    filtered = filtered[
        filtered["customer_id"]
        .astype(str)
        .str.contains(customer_search.strip(), regex=False)
    ]

selected_profile = profile[
    profile["Segment"].isin(selected_segments)
].copy()


# KPI CARDS

total_customers = len(filtered)
total_segments = filtered["Segment"].nunique()

avg_frequency = (
    filtered["Frequency"].mean()
    if not filtered.empty else 0
)

avg_monetary = (
    filtered["Monetary"].mean()
    if not filtered.empty else 0
)

k1, k2, k3, k4 = st.columns(4)

k1.metric("Customers", f"{total_customers:,}")
k2.metric("Segments", f"{total_segments:,}")
k3.metric("Avg. Frequency", f"{avg_frequency:,.2f}")
k4.metric("Avg. Monetary Value", f"{avg_monetary:,.2f}")

st.caption(
    "KPIs reflect the selected filters. Monetary values use "
    "the existing customer-level RFM dataset."
)

st.divider()


# SEGMENT DISTRIBUTION


st.subheader("Customer Distribution by Segment")

if filtered.empty:
    st.info("No customers match the selected filters.")
else:
    distribution = (
        filtered["Segment"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )

    fig = px.bar(
        distribution,
        x="Segment",
        y="Customers",
        color="Segment",
        text="Customers",
        template="plotly_dark",
        title="Customers in Each Business Segment",
    )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        font_color="#E5E7EB",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    st.plotly_chart(fig, use_container_width=True)


# SEGMENT PROFILE

st.subheader("Segment Profile")

if selected_profile.empty:
    st.info("No segment profiles match the selected filters.")
else:
    display_profile = selected_profile[
        required_profile_columns
    ].copy()

    display_profile = display_profile.rename(
        columns={
            "Avg_Recency": "Average Recency",
            "Avg_Frequency": "Average Frequency",
            "Avg_Monetary": "Average Monetary",
            "Avg_RFM_Score": "Average RFM Score",
            "Customer_Percentage": "Customer Share (%)",
        }
    )

    st.dataframe(
        display_profile,
        use_container_width=True,
        hide_index=True,
    )



# RFM ANALYSIS

st.divider()
st.header("RFM Analysis")

st.caption(
    "Understand customer purchasing behaviour using "
    "Recency, Frequency, and Monetary value."
)

if filtered.empty:
    st.info("Select filters that include at least one customer.")
else:

    # 1. RFM CORRELATION HEATMAP

    st.subheader("RFM Feature Correlation")

    rfm_corr = filtered[["Recency", "Frequency", "Monetary"]].corr()

    corr_fig = px.imshow(
        rfm_corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        aspect="auto",
        template="plotly_dark",
        labels={"color": "Correlation"},
    )

    corr_fig.update_layout(
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        font_color="#E5E7EB",
        coloraxis_colorbar_title="Correlation",
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(corr_fig, use_container_width=True,)

    st.caption(
        "Correlation measures the strength and direction of "
        "linear relationships. It does not establish causation."
    )


    # 2. SEGMENT-LEVEL RFM COMPARISON

    st.subheader("Average RFM Metrics by Segment")

    rfm_comparison = (
        filtered.groupby("Segment", as_index=False)
        .agg(
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        )
    )

    col1, col2, col3 = st.columns(3)

    metrics = [
        (col1, "Avg_Recency", "Average Recency (days)"),
        (col2, "Avg_Frequency", "Average Frequency"),
        (col3, "Avg_Monetary", "Average Monetary Value"),
    ]

    for container, metric, title in metrics:
        with container:
            fig = px.bar(
                rfm_comparison.sort_values(metric),
                x=metric,
                y="Segment",
                orientation="h",
                color="Segment",
                text=rfm_comparison.sort_values(metric)[metric].round(2),
                template="plotly_dark",
                title=title,
            )

            fig.update_layout(
                showlegend=False,
                paper_bgcolor="#0B1120",
                plot_bgcolor="#0B1120",
                font_color="#E5E7EB",
                margin=dict(l=10, r=10, t=55, b=20),
            )

            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    # 3. CUSTOMER BEHAVIOUR SCATTER PLOT


    st.subheader("Recency vs Monetary Value")

    scatter_fig = px.scatter(
        filtered,
        x="Recency",
        y="Monetary",
        color="Segment",
        hover_data=[
            "customer_id",
            "Frequency",
            "RFM_Score",
        ],
        opacity=0.65,
        template="plotly_dark",
        labels={
            "Recency": "Recency (days)",
            "Monetary": "Monetary Value",
            "Segment": "Business Segment",
        },
        title="Customer Behaviour by Business Segment",
    )

    # Use a logarithmic monetary axis only when every
    # displayed monetary value is strictly positive.
    if (filtered["Monetary"] > 0).all():
        scatter_fig.update_yaxes(type="log")

    scatter_fig.update_layout(
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        font_color="#E5E7EB",
        legend_title_text="Business Segment",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(scatter_fig, use_container_width=True,)




# PCA VISUALIZATION OF EXISTING ML CLUSTERS
st.divider()
st.header("ML Customer Clusters — PCA")

st.caption(
    "A two-dimensional projection of the saved K-Means "
    "customer assignments. PCA is fitted for visualization; "
    "the existing cluster labels are not retrained."
)

clusters_path = DATA_DIR / "customer_rfm_clusters.parquet"

if not clusters_path.exists():
    st.warning(f"Cluster data not found: {clusters_path}")
else:
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        @st.cache_data(show_spinner="Preparing PCA projection...")
        def prepare_pca_projection(
            file_path: str,
            modified_time: float,
        ):
            del modified_time

            data = pd.read_parquet(file_path)

            required = [
                "customer_id",
                "Recency",
                "Frequency",
                "Monetary",
                "cluster",
                "ML_Segment",
            ]

            missing = [
                col for col in required
                if col not in data.columns
            ]

            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            if data[required].isna().any().any():
                raise ValueError("Cluster dataset contains missing values.")

            if data["customer_id"].duplicated().any():
                raise ValueError("Duplicate customer IDs found.")

            features = data[
                ["Recency", "Frequency", "Monetary"]
            ].copy()

            if (features < 0).any().any():
                raise ValueError("Log1p preprocessing requires non-negative RFM values.")

            # Reproduce the notebook's log transformation.
            log_features = np.log1p(features)

            # Reproduce StandardScaler preprocessing.
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(log_features)

            # Fit PCA for visualization only.
            pca_model = PCA(n_components=2)
            coordinates = pca_model.fit_transform(scaled_features)

            plot_data = pd.DataFrame({
                "customer_id": data["customer_id"].values,
                "PC1": coordinates[:, 0],
                "PC2": coordinates[:, 1],
                "cluster": data["cluster"].values,
                "ML_Segment": data["ML_Segment"].values,
            })

            variance = pca_model.explained_variance_ratio_

            return plot_data, variance

        pca_data, explained_variance = prepare_pca_projection(
            str(clusters_path),
            clusters_path.stat().st_mtime,
        )

        pc1_variance = explained_variance[0] * 100
        pc2_variance = explained_variance[1] * 100
        total_variance = explained_variance.sum() * 100

        k1, k2, k3 = st.columns(3)

        k1.metric("PC1 Explained Variance", f"{pc1_variance:.2f}%")
        k2.metric("PC2 Explained Variance", f"{pc2_variance:.2f}%")
        k3.metric("Total Explained Variance", f"{total_variance:.2f}%")

        st.caption(
            "Explained variance indicates how much of the standardized "
            "RFM feature variation is represented by these two components."
        )

        cluster_order = (
            pca_data[["cluster", "ML_Segment"]]
            .drop_duplicates()
            .sort_values("cluster")
        )

        selected_ml_segments = st.multiselect(
            "Filter ML segments",
            options=cluster_order["ML_Segment"].tolist(),
            default=cluster_order["ML_Segment"].tolist(),
            key="pca_ml_segment_filter",
        )

        plot_data = pca_data[
            pca_data["ML_Segment"].isin(selected_ml_segments)
        ].copy()

        if plot_data.empty:
            st.info("Select at least one ML segment to display the PCA plot.")
        else:
            pca_fig = px.scatter(
                plot_data,
                x="PC1",
                y="PC2",
                color="ML_Segment",
                hover_data=[
                    "customer_id",
                    "cluster",
                ],
                opacity=0.65,
                template="plotly_dark",
                title="Existing K-Means Assignments in PCA Space",
                labels={
                    "PC1": f"PC1 ({pc1_variance:.1f}% variance)",
                    "PC2": f"PC2 ({pc2_variance:.1f}% variance)",
                    "ML_Segment": "ML Segment",
                },
            )

            pca_fig.update_layout(
                paper_bgcolor="#0B1120",
                plot_bgcolor="#0B1120",
                font_color="#E5E7EB",
                margin=dict(l=20, r=20, t=60, b=20),
            )

            st.plotly_chart(pca_fig, use_container_width=True,)

            st.download_button(
                "Download PCA Coordinates",
                data=plot_data.to_csv(index=False).encode("utf-8"),
                file_name="retailpulse_pca_coordinates.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Could not prepare the PCA visualization: {exc}")



# BUSINESS SEGMENTS VS ML CLUSTERS

st.divider()
st.header("Business Segments vs ML Clusters")

st.caption(
    "Compare rule-based RFM business segments with "
    "the saved K-Means customer classifications."
)

comparison_path = DATA_DIR / "rfm_kmeans_comparison.csv"

if not comparison_path.exists():
    st.warning(f"Comparison file not found: {comparison_path}")
else:
    try:
        comparison_df = pd.read_csv(comparison_path)

        required_columns = [
            "Segment",
            "At Risk",
            "Champions",
            "Lost / Hibernating",
            "Loyal / Promising Customers",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in comparison_df.columns
        ]

        if missing_columns:
            st.error(f"Missing comparison columns: {missing_columns}")
        elif comparison_df["Segment"].duplicated().any():
            st.error("Duplicate business segment labels in comparison file.")
        else:
            matrix = comparison_df.set_index("Segment")[required_columns[1:]]

            if (
                matrix.isna().any().any()
                or (matrix < 0).any().any()
            ):
                st.error(
                    "The comparison matrix contains missing "
                    "or negative customer counts."
                )
            else:
                col1, col2 = st.columns([1, 2])

                with col1:
                    view_mode = st.selectbox(
                        "Heatmap display",
                        [
                            "Customer counts",
                            "Row percentages",
                        ],
                    )

                with col2:
                    st.metric(
                        "Customers represented",
                        f"{int(matrix.to_numpy().sum()):,}",
                    )

                if view_mode == "Row percentages":
                    row_totals = matrix.sum(axis=1)

                    if (row_totals == 0).any():
                        st.error(
                            "At least one business segment has "
                            "zero customers; percentages cannot "
                            "be calculated for that segment."
                        )
                        st.stop()

                    plot_matrix = matrix.div(
                        row_totals,
                        axis=0,
                    ) * 100

                    text_format = ".1f"
                    colorbar_title = "Customers (%)"
                    zmin = 0
                    zmax = 100

                else:
                    plot_matrix = matrix
                    text_format = ",.0f"
                    colorbar_title = "Customers"
                    zmin = 0
                    zmax = max(
                        1,
                        float(matrix.to_numpy().max()),
                    )

                heatmap = px.imshow(
                    plot_matrix,
                    x=plot_matrix.columns.tolist(),
                    y=plot_matrix.index.tolist(),
                    text_auto=text_format,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    zmin=zmin,
                    zmax=zmax,
                    labels={
                        "x": "ML Segment",
                        "y": "Business Segment",
                        "color": colorbar_title,
                    },
                    template="plotly_dark",
                )

                heatmap.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#0B1120",
                    font_color="#E5E7EB",
                    xaxis_side="bottom",
                    margin=dict(
                        l=20,
                        r=20,
                        t=35,
                        b=100,
                    ),
                )

                heatmap.update_xaxes(tickangle=-20,)

                st.plotly_chart(heatmap,use_container_width=True,)

                st.caption(
                    "Rows represent business-defined segments; "
                    "columns represent saved ML segment assignments. "
                    "Row percentages describe the distribution within "
                    "each business segment, not overall model accuracy."
                )

    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not load the segmentation comparison: {exc}")



# CUSTOMER EXPLORER

st.subheader("Customer Explorer")
st.caption(
    "Inspect individual customers and download the filtered results."
)

display_columns = [
    "customer_id",
    "Segment",
    "Recency",
    "Frequency",
    "Monetary",
    "R_Score",
    "F_Score",
    "M_Score",
    "RFM_Score",
]

st.write(f"Matching customers: {len(filtered):,}")

st.dataframe(
    filtered[display_columns].sort_values(
        "Monetary",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    label="Download Customer Segment Report",
    data=filtered[display_columns].to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_customer_segments.csv",
    mime="text/csv",
)

st.caption("Source: saved customer RFM segmentation results and segment profiles.")
