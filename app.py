
import streamlit as st

st.set_page_config(
    page_title="RetailPulse | Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: #0B1120;
}
[data-testid="stSidebar"] {
    background: #151F32;
}
[data-testid="stMetric"] {
    background: #151F32;
    border: 1px solid #2A3B56;
    padding: 16px 18px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 RetailPulse")
st.subheader("Retail Business Intelligence Platform")

st.write(
    "Explore customer behavior, demand patterns, inventory signals, and churn risk "
    "from a unified analytics workspace. Choose a module from the sidebar to begin."
)

st.divider()

cards = [
    (
        "👥 Customer Segmentation",
        "Understand customer groups and purchasing behavior.",
    ),
    (
        "📈 Demand Forecasting",
        "Explore historical demand and forecast future sales.",
    ),
    (
        "📦 Inventory Management",
        "Review product movement, demand trends, and inventory priorities.",
    ),
    (
        "🎯 Churn Prediction",
        "Inspect churn risk, retention priorities, and customer explanations.",
    ),
]

cols = st.columns(2)

for i, (title, description) in enumerate(cards):
    with cols[i % 2]:
        with st.container(border=True):
            st.subheader(title)
            st.write(description)
            st.caption("Use the page navigation in the sidebar to open this module.")

st.divider()
st.caption("RetailPulse • Customer, demand, inventory, and retention intelligence")
