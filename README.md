# RetailPulse - Retail Analytics & Intelligence Dashboard

RetailPulse is an end-to-end retail analytics project built with Python and Streamlit. It brings together customer segmentation, demand forecasting, inventory optimization, and customer churn analysis in one interactive dashboard.

The project demonstrates data preparation, exploratory analysis, machine learning workflows, forecasting evaluation, and data-driven retail decision support.

## Features

### 1. Customer Segmentation
- Analyze customer purchasing behavior using RFM (Recency, Frequency, Monetary) analysis.
- Explore customer segments and segment profiles.
- Visualize clustering results and compare segmentation approaches.
- Export customer segmentation results for further analysis.

### 2. Demand Forecasting
- Explore historical retail demand and forecast results.
- Compare Prophet-based and hybrid forecasting predictions.
- Review forecasting performance metrics and model comparisons.
- Analyze forecast errors and cross-validation results when the corresponding artifacts are available.

### 3. Inventory Optimization
- Classify products by movement and demand behavior.
- Identify products requiring inventory review.
- Analyze demand trends, variability, and recent sales patterns.
- Investigate product-level demand history and transaction details.
- Download filtered product data and investigation reports.

### 4. Churn Prediction
- Explore customer churn risk and retention priorities.
- Review customer risk summaries and prediction results.
- Identify customers who may need retention-focused attention.
- Export retention reports for further analysis.

## Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Data processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Machine learning:** Scikit-learn
- **Data storage:** CSV and Parquet
- **Development environment:** Jupyter Notebook, VS Code

## Project Structure

```text
RetailPulse/
├── app.py
├── pages/
│   ├── 1_Customer_Segmentation.py
│   ├── 2_Demand_Forecasting.py
│   ├── 3_Inventory_Management.py
│   └── 4_Churn_Prediction.py
├── data/
│   └── processed/
├── notebooks/
│   ├── 01_Initial_EDA.ipynb
│   ├── 02_Demand_Forecasting.ipynb
│   ├── 03_Customer_Segmentation.ipynb
│   ├── 04_Churn_Prediction.ipynb
│   └── 05_Inventory_Optimization.ipynb
├── outputs/
├── .streamlit/
│   └── config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

The processed datasets and output artifacts support the dashboard pages. Some optional analytical sections depend on additional files generated during the analysis workflows.

## Getting Started

### Prerequisites

- Python 3.11
- Git
- A terminal or command prompt

### 1. Clone the repository

```bash
git clone https://github.com/Rushankparab1234/retailpulse.git
cd retailpulse
```

### 2. Create a virtual environment

On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to explore the application.

## Data and Reproducibility

RetailPulse uses processed CSV and Parquet datasets, forecasting metrics, and analytical outputs to power its dashboards.

The original raw dataset is excluded from the public repository configuration. To reproduce the analysis from raw data, obtain the dataset from an authorized source and follow the preparation steps documented in the notebooks.

Some dashboard sections rely on precomputed results rather than retraining models when the application starts.

## Current Scope and Limitations

- Forecasting and historical analysis depend on the available dataset and precomputed artifacts.
- Dashboard sections that rely on optional reports may display less information if those files are unavailable.
- Forecasts and churn scores are analytical estimates and should not be treated as guaranteed outcomes.
- Local execution and cloud deployment can differ because of dependency versions and resource limits.

## Future Improvements

- Deploy the application using Streamlit Community Cloud.
- Add automated data-quality and deployment checks.
- Improve model monitoring and forecasting evaluation.
- Introduce configurable inventory policies and reorder recommendations.
- Add clearer explanations of customer risk and model predictions.

## Author

**Rushank Parab**

GitHub: [Rushankparab1234](https://github.com/Rushankparab1234)

---

*RetailPulse — Turning retail data into actionable insights.*
