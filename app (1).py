import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales & Demand Forecasting Dashboard")
st.markdown("### Machine Learning Task 1 (2026)")

uploaded_file = st.file_uploader(
    "Upload Superstore Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, encoding='latin1')

    st.success("Dataset Uploaded Successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --------------------
    # DATE CONVERSION
    # --------------------

    df['Order Date'] = pd.to_datetime(
        df['Order Date'],
        errors='coerce'
    )

    # --------------------
    # KPI CARDS
    # --------------------

    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    total_orders = df['Order ID'].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    col2.metric(
        "📊 Total Profit",
        f"${total_profit:,.0f}"
    )

    col3.metric(
        "📦 Total Orders",
        total_orders
    )

    st.divider()

    # --------------------
    # MONTHLY SALES TREND
    # --------------------

    df['Month'] = df['Order Date'].dt.to_period('M')

    monthly_sales = (
        df.groupby('Month')['Sales']
        .sum()
        .reset_index()
    )

    monthly_sales['Month'] = monthly_sales['Month'].astype(str)

    st.subheader("📈 Monthly Sales Trend")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(
        monthly_sales['Month'],
        monthly_sales['Sales'],
        marker='o'
    )
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # --------------------
    # REGION SALES
    # --------------------

    st.subheader("🌍 Region Wise Sales")

    region_sales = (
        df.groupby('Region')['Sales']
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    region_sales.plot(
        kind='bar',
        ax=ax
    )

    st.pyplot(fig)

    # --------------------
    # CATEGORY SALES
    # --------------------

    st.subheader("🛒 Category Wise Sales")

    category_sales = (
        df.groupby('Category')['Sales']
        .sum()
    )

    fig, ax = plt.subplots()

    category_sales.plot(
        kind='pie',
        autopct='%1.1f%%',
        ax=ax
    )

    st.pyplot(fig)

    # --------------------
    # TOP PRODUCTS
    # --------------------

    st.subheader("🏆 Top 10 Products")

    top_products = (
        df.groupby('Product Name')['Sales']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.dataframe(top_products)

    # --------------------
    # MACHINE LEARNING
    # --------------------

    st.header("🤖 Sales Forecasting")

    monthly_model = (
        df.groupby('Month')['Sales']
        .sum()
        .reset_index()
    )

    monthly_model['Month_Num'] = np.arange(
        len(monthly_model)
    )

    X = monthly_model[['Month_Num']]
    y = monthly_model['Sales']

    split = int(len(monthly_model)*0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        round(mae,2)
    )

    col2.metric(
        "RMSE",
        round(rmse,2)
    )

    # --------------------
    # FUTURE FORECAST
    # --------------------

    future_months = 12

    future_x = pd.DataFrame({
        'Month_Num':
        np.arange(
            len(monthly_model),
            len(monthly_model)+future_months
        )
    })

    future_pred = model.predict(
        future_x
    )

    forecast_df = pd.DataFrame({
        'Future Month':
        np.arange(
            1,
            future_months+1
        ),
        'Predicted Sales':
        future_pred
    })

    st.subheader("🔮 Future Sales Forecast")

    st.dataframe(forecast_df)

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        forecast_df['Future Month'],
        forecast_df['Predicted Sales'],
        marker='o'
    )

    ax.set_title(
        "Next 12 Months Forecast"
    )

    st.pyplot(fig)

    st.success(
        "Forecast Generated Successfully!"
    )
