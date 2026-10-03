import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Pension Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Load model & dataset (cached)
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("twopot_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("pension_finance_risk_dataset.csv")

model = load_model()
df = load_data()

# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
st.sidebar.title("📊 Pension Risk Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Predict Risk", "📈 Data Explorer", "📊 Model Insights"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Two-Pot Pension Model • Powered by Random Forest")

# -------------------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🏠 Pension Fund Risk Assessment Dashboard")
    st.markdown(
        """
        Welcome! This app predicts the **Pension Risk Level**
        (`Low Risk`, `Medium Risk`, `High Risk`) based on fund and
        macroeconomic indicators.

        ### How to use
        1. Go to **🔮 Predict Risk** to enter features and get a prediction.
        2. Explore the dataset under **📈 Data Explorer**.
        3. Check the model overview under **📊 Model Insights**.

        ### Dataset at a glance
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Features (model)", 35)
    col4.metric("Target Classes", df["Pension_Risk_Level"].nunique())

    st.subheader("Risk Level Distribution")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    df["Pension_Risk_Level"].value_counts().plot(
        kind="bar", color=["#2ecc71", "#f39c12", "#e74c3c"], ax=ax
    )
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# -------------------------------------------------------------------
# PREDICT PAGE
# -------------------------------------------------------------------
elif page == "🔮 Predict Risk":
    st.title("🔮 Predict Pension Risk Level")
    st.markdown("Adjust the inputs below and click **Predict**.")

    # ---- Model feature columns (from model_features.pkl) ----
    NUMERIC_FEATURES = [
        "Average_Age", "Life_Expectancy", "Contribution_Rate",
        "Withdrawal_Rate", "Fund_Size_Million", "Market_Return",
        "Inflation_Rate", "Interest_Rate", "Stock_Volatility",
        "GDP_Growth", "Unemployment_Rate", "Dependency_Ratio",
        "Equity_Allocation", "Bond_Allocation", "Alternative_Investment",
        "Open_Price", "Close_Price", "Daily_High", "Daily_Low",
        "Trading_Volume", "GDP_Growth_(%)", "Inflation_Rate_(%)",
        "Unemployment_Rate_(%)", "Interest_Rate_(%)",
        "Government_Debt_(Billion_USD)", "Corporate_Profits_(Billion_USD)",
        "Forex_USD/EUR", "Forex_USD/JPY",
        "Crude_Oil_Price_(USD_per_Barrel)", "Gold_Price_(USD_per_Ounce)",
        "Retail_Sales_(Billion_USD)", "Bankruptcy_Rate_(%)",
        "Mergers_&_Acquisitions_Deals",
        "Venture_Capital_Funding_(Billion_USD)",
    ]

    # Sensible defaults = dataset means (or 0 if not present)
    defaults = {}
    for f in NUMERIC_FEATURES:
        if f in df.columns:
            defaults[f] = float(df[f].mean())
        else:
            defaults[f] = 0.0

    # ---------- Input form ----------
    with st.form("prediction_form"):
        st.subheader("👤 Demographic / Fund Profile")
        c1, c2, c3 = st.columns(3)
        avg_age = c1.number_input("Average Age", 18.0, 100.0, defaults["Average_Age"])
        life_exp = c2.number_input("Life Expectancy", 40.0, 120.0, defaults["Life_Expectancy"])
        contrib = c3.number_input("Contribution Rate (%)", 0.0, 100.0, defaults["Contribution_Rate"])

        c1, c2, c3 = st.columns(3)
        withd = c1.number_input("Withdrawal Rate (%)", 0.0, 100.0, defaults["Withdrawal_Rate"])
        fund_size = c2.number_input("Fund Size (Million)", 0.0, 1e6, defaults["Fund_Size_Million"])
        mkt_ret = c3.number_input("Market Return (%)", -50.0, 100.0, defaults["Market_Return"])

        st.subheader("📉 Macroeconomic Indicators")
        c1, c2, c3 = st.columns(3)
        infl = c1.number_input("Inflation Rate (%)", -10.0, 100.0, defaults["Inflation_Rate"])
        int_rate = c2.number_input("Interest Rate (%)", 0.0, 50.0, defaults["Interest_Rate"])
        vol = c3.number_input("Stock Volatility", 0.0, 200.0, defaults["Stock_Volatility"])

        c1, c2, c3 = st.columns(3)
        gdp = c1.number_input("GDP Growth (%)", -20.0, 50.0, defaults["GDP_Growth"])
        unemp = c2.number_input("Unemployment Rate (%)", 0.0, 50.0, defaults["Unemployment_Rate"])
        dep = c3.number_input("Dependency Ratio", 0.0, 2.0, defaults["Dependency_Ratio"])

        st.subheader("💼 Asset Allocation")
        c1, c2, c3 = st.columns(3)
        eq = c1.slider("Equity Allocation (%)", 0.0, 100.0, float(defaults["Equity_Allocation"]))
        bd = c2.slider("Bond Allocation (%)", 0.0, 100.0, float(defaults["Bond_Allocation"]))
        alt = c3.slider("Alternative Investment (%)", 0.0, 100.0, float(defaults["Alternative_Investment"]))

        with st.expander("📊 Market Data (optional)"):
            c1, c2, c3 = st.columns(3)
            open_p = c1.number_input("Open Price", 0.0, value=defaults["Open_Price"])
            close_p = c2.number_input("Close Price", 0.0, value=defaults["Close_Price"])
            high_p = c3.number_input("Daily High", 0.0, value=defaults["Daily_High"])

            c1, c2, c3 = st.columns(3)
            low_p = c1.number_input("Daily Low", 0.0, value=defaults["Daily_Low"])
            vol_ = c2.number_input("Trading Volume", 0.0, value=defaults["Trading_Volume"])
            gdp_pct = c3.number_input("GDP Growth (%)", value=defaults["GDP_Growth_(%)"])

            c1, c2, c3 = st.columns(3)
            inf_pct = c1.number_input("Inflation Rate (%)", value=defaults["Inflation_Rate_(%)"])
            une_pct = c2.number_input("Unemployment Rate (%)", value=defaults["Unemployment_Rate_(%)"])
            intr_pct = c3.number_input("Interest Rate (%)", value=defaults["Interest_Rate_(%)"])

            c1, c2, c3 = st.columns(3)
            debt = c1.number_input("Government Debt (B USD)", value=defaults["Government_Debt_(Billion_USD)"])
            profit = c2.number_input("Corporate Profits (B USD)", value=defaults["Corporate_Profits_(Billion_USD)"])
            fx_eur = c3.number_input("Forex USD/EUR", value=defaults["Forex_USD/EUR"])

            c1, c2, c3 = st.columns(3)
            fx_jpy = c1.number_input("Forex USD/JPY", value=defaults["Forex_USD/JPY"])
            oil = c2.number_input("Crude Oil (USD/bbl)", value=defaults["Crude_Oil_Price_(USD_per_Barrel)"])
            gold = c3.number_input("Gold (USD/oz)", value=defaults["Gold_Price_(USD_per_Ounce)"])

            c1, c2, c3 = st.columns(3)
            retail = c1.number_input("Retail Sales (B USD)", value=defaults["Retail_Sales_(Billion_USD)"])
            bank = c2.number_input("Bankruptcy Rate (%)", value=defaults["Bankruptcy_Rate_(%)"])
            ma = c3.number_input("M&A Deals", value=defaults["Mergers_&_Acquisitions_Deals"])

            vc = st.number_input("VC Funding (B USD)", value=defaults["Venture_Capital_Funding_(Billion_USD)"])

        submitted = st.form_submit_button("🚀 Predict Risk Level", use_container_width=True)

    # ---------- Build DataFrame & Predict ----------
    if submitted:
        input_dict = {
            "Average_Age": avg_age,
            "Life_Expectancy": life_exp,
            "Contribution_Rate": contrib,
            "Withdrawal_Rate": withd,
            "Fund_Size_Million": fund_size,
            "Market_Return": mkt_ret,
            "Inflation_Rate": infl,
            "Interest_Rate": int_rate,
            "Stock_Volatility": vol,
            "GDP_Growth": gdp,
            "Unemployment_Rate": unemp,
            "Dependency_Ratio": dep,
            "Equity_Allocation": eq,
            "Bond_Allocation": bd,
            "Alternative_Investment": alt,
            "Open_Price": open_p,
            "Close_Price": close_p,
            "Daily_High": high_p,
            "Daily_Low": low_p,
            "Trading_Volume": vol_,
            "GDP_Growth_(%)": gdp_pct,
            "Inflation_Rate_(%)": inf_pct,
            "Unemployment_Rate_(%)": une_pct,
            "Interest_Rate_(%)": intr_pct,
            "Government_Debt_(Billion_USD)": debt,
            "Corporate_Profits_(Billion_USD)": profit,
            "Forex_USD/EUR": fx_eur,
            "Forex_USD/JPY": fx_jpy,
            "Crude_Oil_Price_(USD_per_Barrel)": oil,
            "Gold_Price_(USD_per_Ounce)": gold,
            "Retail_Sales_(Billion_USD)": retail,
            "Bankruptcy_Rate_(%)": bank,
            "Mergers_&_Acquisitions_Deals": ma,
            "Venture_Capital_Funding_(Billion_USD)": vc,
        }

        input_df = pd.DataFrame([input_dict])

        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_

            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            color_map = {
                "Low Risk": "🟢",
                "Medium Risk": "🟡",
                "High Risk": "🔴",
            }
            st.markdown(
                f"<h2 style='text-align:center'>{color_map.get(pred,'')} "
                f"<b>{pred}</b></h2>",
                unsafe_allow_html=True,
            )

            # Probability bars
            prob_df = pd.DataFrame(
                {"Risk Level": classes, "Probability": proba}
            ).sort_values("Probability", ascending=False)

            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(
                data=prob_df, x="Probability", y="Risk Level",
                palette={"Low Risk": "#2ecc71", "Medium Risk": "#f39c12",
                         "High Risk": "#e74c3c"},
                ax=ax,
            )
            ax.set_xlim(0, 1)
            for i, v in enumerate(prob_df["Probability"]):
                ax.text(v + 0.01, i, f"{v:.1%}", va="center")
            st.pyplot(fig)

            st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------------------------------------------
# DATA EXPLORER
# -------------------------------------------------------------------
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")

    tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Correlation"])

    with tab1:
        st.subheader("Dataset Preview")
        n = st.slider("Rows to show", 5, 100, 15)
        st.dataframe(df.head(n), use_container_width=True)

    with tab2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T, use_container_width=True)

        st.subheader("Risk Level Counts")
        st.dataframe(
            df["Pension_Risk_Level"].value_counts().rename("Count"),
            use_container_width=True,
        )

    with tab3:
        st.subheader("Correlation Heatmap (numeric)")
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(12, 9))
            sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns.")

# -------------------------------------------------------------------
# MODEL INSIGHTS
# -------------------------------------------------------------------
elif page == "📊 Model Insights":
    st.title("📊 Model Insights")
    st.markdown(
        """
        This app uses a **Pipeline** (preprocessing + `RandomForestRegressor`)
        stored in `twopot_model.pkl`.

        The preprocessing step:
        * Imputes missing numeric values with the **median** and scales them.
        * One-hot encodes `Pension_Risk_Level` (target).

        The regressor is a **Random Forest** with 100 trees.
        """
    )

    # Feature importance (if available)
    try:
        pre = model.named_steps["preprocessor"]
        reg = model.named_steps["regressor"]

        # Extract feature names after preprocessing
        num_cols = list(pre.transformers_[0][2])
        cat_cols = list(pre.transformers_[1][2])

        ohe = pre.transformers_[1][1].named_steps["encoder"]
        cat_names = list(ohe.get_feature_names_out(cat_cols))

        feature_names = num_cols + cat_names
        importances = reg.feature_importances_

        fi = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(20)
        )

        st.subheader("Top 20 Feature Importances")
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.barplot(data=fi, x="Importance", y="Feature", palette="viridis", ax=ax)
        st.pyplot(fig)

        st.dataframe(fi, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")

    st.subheader("Model Pipeline")
    st.code(str(model))