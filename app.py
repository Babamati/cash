import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Treasury Cash Forecasting System",
    page_icon="📈",
    layout="wide"
)

# -------------------------
# Load precomputed outputs
# -------------------------
@st.cache_data
def load_outputs():
    forecast_df = pd.read_excel("outputs/weekly_expenditure_hybrid_forecast_52_weeks.xlsx")
    test_results_df = pd.read_excel("outputs/weekly_expenditure_hybrid_test_predictions.xlsx")
    importance_df = pd.read_excel("outputs/weekly_expenditure_feature_importance.xlsx")
    comparison_df = pd.read_excel("outputs/weekly_expenditure_model_vs_baselines.xlsx")

    # Ensure date columns are datetime
    if "Date" in forecast_df.columns:
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    if "Date" in test_results_df.columns:
        test_results_df["Date"] = pd.to_datetime(test_results_df["Date"])

    return forecast_df, test_results_df, importance_df, comparison_df


def plot_actual_vs_predicted(test_results_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_results_df["Date"], test_results_df["Actual"], label="Actual")
    ax.plot(test_results_df["Date"], test_results_df["Predicted_Hybrid"], label="Hybrid Predicted")

    if "Lower_95" in test_results_df.columns and "Upper_95" in test_results_df.columns:
        ax.fill_between(
            test_results_df["Date"],
            test_results_df["Lower_95"],
            test_results_df["Upper_95"],
            alpha=0.2,
            label="Confidence Interval"
        )

    ax.set_title("Actual vs Predicted Weekly Expenditure")
    ax.set_xlabel("Week Ending")
    ax.set_ylabel("Weekly Expenditure")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_baseline_comparison(test_results_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_results_df["Date"], test_results_df["Actual"], label="Actual")
    ax.plot(test_results_df["Date"], test_results_df["Predicted_Hybrid"], label="Hybrid Predicted")
    ax.plot(test_results_df["Date"], test_results_df["Baseline_Fixed_100k"], "--", label="Baseline 100k")
    ax.plot(test_results_df["Date"], test_results_df["Baseline_Train_Mean"], "--", label="Baseline Train Mean")
    ax.plot(test_results_df["Date"], test_results_df["Baseline_Last_Value"], "--", label="Baseline Last Value")
    ax.plot(test_results_df["Date"], test_results_df["Baseline_Seasonal_Naive"], "--", label="Baseline Seasonal Naive")

    ax.set_title("Model Performance Against Baselines")
    ax.set_xlabel("Week Ending")
    ax.set_ylabel("Weekly Expenditure")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_future_forecast(forecast_df, test_results_df):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Show recent actuals if available
    if "Date" in test_results_df.columns and "Actual" in test_results_df.columns:
        ax.plot(
            test_results_df["Date"].tail(20),
            test_results_df["Actual"].tail(20),
            label="Recent Actuals"
        )

    ax.plot(
        forecast_df["Date"],
        forecast_df["Predicted_Weekly_Expenditure"],
        label="52-Week Forecast"
    )

    if "Lower_95" in forecast_df.columns and "Upper_95" in forecast_df.columns:
        ax.fill_between(
            forecast_df["Date"],
            forecast_df["Lower_95"],
            forecast_df["Upper_95"],
            alpha=0.2,
            label="Confidence Interval"
        )

    ax.set_title("Projected Weekly Expenditure")
    ax.set_xlabel("Week Ending")
    ax.set_ylabel("Weekly Expenditure")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df):
    top_df = importance_df.head(15).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["Feature"], top_df["Importance"])
    ax.set_title("Top Predictive Features")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


# -------------------------
# App UI
# -------------------------
st.title("📈 Treasury Cash Forecasting System")
st.markdown(
    """
    This application presents precomputed forecasting results from a hybrid
    machine learning and time series model for weekly expenditure forecasting.

    It is designed as a portfolio demonstration showing:
    - model performance against actuals
    - comparison with baseline methods
    - 52-week forward forecast
    - feature importance analysis
    """
)

try:
    forecast_df, test_results_df, importance_df, comparison_df = load_outputs()

    hybrid_row = comparison_df[comparison_df["Model"] == "Hybrid Model"].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{hybrid_row['MAE']:,.2f}")
    col2.metric("RMSE", f"{hybrid_row['RMSE']:,.2f}")
    col3.metric("R²", f"{hybrid_row['R2']:.4f}")

    st.subheader("1. Actual vs Predicted")
    st.pyplot(plot_actual_vs_predicted(test_results_df))

    st.subheader("2. Baseline Comparison")
    st.pyplot(plot_baseline_comparison(test_results_df))

    st.subheader("3. Forward Forecast")
    st.pyplot(plot_future_forecast(forecast_df, test_results_df))

    st.subheader("4. Feature Importance")
    st.pyplot(plot_feature_importance(importance_df))

    with st.expander("Model Comparison Table"):
        st.dataframe(comparison_df, use_container_width=True)

    with st.expander("Forecast Output Table"):
        st.dataframe(forecast_df, use_container_width=True)

    with st.expander("Feature Importance Table"):
        st.dataframe(importance_df, use_container_width=True)

except FileNotFoundError:
    st.error("One or more output files were not found in the outputs folder.")
except Exception as e:
    st.error(f"Error loading dashboard: {e}")
