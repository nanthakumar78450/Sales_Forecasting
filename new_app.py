import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_FILE = "final_catboost_sales_model.pkl"

DATA_FILE = "Sales_Forcasting_Dataset_Corrected.xlsx"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    return model


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    data = pd.read_excel(
        DATA_FILE
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.sort_values(
        "Date"
    ).reset_index(drop=True)

    return data


# =========================================================
# LOAD MODEL + DATA
# =========================================================

try:

    model = load_model()

    df = load_dataset()

except FileNotFoundError as e:

    st.error(
        "❌ Required file not found!"
    )

    st.info(
        """
Make sure these files are in the same folder:

app.py
final_catboost_sales_model.pkl
Sales_Forcasting_Dataset_Corrected.xlsx
        """
    )

    st.exception(e)

    st.stop()

except Exception as e:

    st.error(
        "❌ Application loading error!"
    )

    st.exception(e)

    st.stop()


# =========================================================
# GET MODEL FEATURE NAMES
# =========================================================

if hasattr(model, "feature_names_"):

    FEATURE_COLUMNS = list(
        model.feature_names_
    )

else:

    st.error(
        "❌ CatBoost model does not contain feature names."
    )

    st.stop()


# =========================================================
# PRODUCT OPTIONS
# =========================================================

product_data = (
    df[
        [
            "Product_ID",
            "Product_Name"
        ]
    ]
    .drop_duplicates()
    .sort_values("Product_ID")
)


product_options = product_data[
    "Product_Name"
].tolist()


# =========================================================
# STORE OPTIONS
# =========================================================

store_data = (
    df[
        [
            "Store_ID",
            "Store_Location"
        ]
    ]
    .drop_duplicates()
    .sort_values("Store_ID")
)


store_options = store_data[
    "Store_ID"
].tolist()


# =========================================================
# TITLE
# =========================================================

st.title(
    "📈 Sales Prediction System"
)

st.write(
    "CatBoost Based Units Sold Prediction"
)

st.success(
    "🏆 Best Model: CatBoost Regressor"
)

st.markdown("---")


# =========================================================
# INPUT SECTION
# =========================================================

st.header(
    "📝 Enter Sales Details"
)


col1, col2 = st.columns(2)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    forecast_date = st.date_input(
        "📅 Forecast Date",
        value=pd.Timestamp.today().date()
    )


    product = st.selectbox(
        "📦 Product",
        product_options
    )


    store = st.selectbox(
        "🏪 Store",
        store_options
    )


    price = st.number_input(
        "💰 Price",
        min_value=0.0,
        value=1000.0,
        step=1.0
    )


    discount = st.number_input(
        "🏷️ Discount %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0
    )


    promotion = st.selectbox(
        "📢 Promotion",
        [
            "Yes",
            "No"
        ]
    )


    stock = st.number_input(
        "📦 Stock Availability",
        min_value=0.0,
        value=100.0,
        step=1.0
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    holiday = st.selectbox(
        "🎉 Holiday",
        [
            "Yes",
            "No"
        ]
    )


    local_event = st.selectbox(
        "📍 Local Event",
        [
            "Yes",
            "No"
        ]
    )


    competitor_price = st.number_input(
        "💰 Competitor Price",
        min_value=0.0,
        value=1000.0,
        step=1.0
    )


    economic_indicator = st.number_input(
        "📊 Economic Indicator",
        value=100.0,
        step=0.1
    )


    marketing_spend = st.number_input(
        "📣 Marketing Spend",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )


    forecast_horizon = st.selectbox(
        "🔮 Forecast Horizon",
        [
            "1 Day",
            "7 Days",
            "30 Days",
            "90 Days",
            "1 Year",
            "2 Years"
        ]
    )


# =========================================================
# HORIZON MAPPING
# =========================================================

horizon_map = {

    "1 Day": 1,

    "7 Days": 7,

    "30 Days": 30,

    "90 Days": 90,

    "1 Year": 365,

    "2 Years": 730

}


forecast_days = horizon_map[
    forecast_horizon
]


# =========================================================
# BINARY FEATURES
# =========================================================

promotion_flag = (
    1 if promotion == "Yes"
    else 0
)

holiday_flag = (
    1 if holiday == "Yes"
    else 0
)

local_event_flag = (
    1 if local_event == "Yes"
    else 0
)


# =========================================================
# FIND PRODUCT ID
# =========================================================

selected_product_id = (
    product_data.loc[
        product_data["Product_Name"] == product,
        "Product_ID"
    ]
    .iloc[0]
)


# =========================================================
# FIND STORE LOCATION
# =========================================================

selected_store_location = (
    store_data.loc[
        store_data["Store_ID"] == store,
        "Store_Location"
    ]
    .iloc[0]
)


# =========================================================
# BUILD FUTURE ROW
# =========================================================

def build_future_row(
    date,
    product_id,
    store_id,
    product_name,
    store_location,
    history
):

    date = pd.Timestamp(
        date
    )


    # -----------------------------------------------------
    # DATE FEATURES
    # -----------------------------------------------------

    year = date.year

    month = date.month

    day = date.day

    day_of_week = date.dayofweek

    quarter = date.quarter

    week_of_year = int(
        date.isocalendar().week
    )


    # -----------------------------------------------------
    # WEEK FEATURES
    # -----------------------------------------------------

    is_weekend = int(
        day_of_week >= 5
    )

    is_weekday = int(
        day_of_week < 5
    )


    # -----------------------------------------------------
    # CYCLICAL FEATURES
    # -----------------------------------------------------

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    dayofweek_sin = np.sin(
        2 * np.pi * day_of_week / 7
    )

    dayofweek_cos = np.cos(
        2 * np.pi * day_of_week / 7
    )


    # -----------------------------------------------------
    # HISTORY
    # -----------------------------------------------------

    hist = list(
        history
    )


    # -----------------------------------------------------
    # LAG FEATURES
    # -----------------------------------------------------

    lag_1 = (
        hist[-1]
        if len(hist) >= 1
        else 0
    )


    lag_7 = (
        hist[-7]
        if len(hist) >= 7
        else lag_1
    )


    lag_14 = (
        hist[-14]
        if len(hist) >= 14
        else lag_1
    )


    lag_30 = (
        hist[-30]
        if len(hist) >= 30
        else lag_1
    )


    # -----------------------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------------------

    last7 = hist[-7:]

    last14 = hist[-14:]

    last30 = hist[-30:]


    if len(last7) > 0:

        rolling_mean_7 = np.mean(
            last7
        )

        rolling_max_7 = np.max(
            last7
        )

        rolling_min_7 = np.min(
            last7
        )

    else:

        rolling_mean_7 = 0

        rolling_max_7 = 0

        rolling_min_7 = 0


    if len(last14) > 0:

        rolling_mean_14 = np.mean(
            last14
        )

    else:

        rolling_mean_14 = 0


    if len(last30) > 0:

        rolling_mean_30 = np.mean(
            last30
        )

    else:

        rolling_mean_30 = 0


    if len(last7) > 1:

        rolling_std_7 = np.std(
            last7,
            ddof=1
        )

    else:

        rolling_std_7 = 0


    # -----------------------------------------------------
    # EXPANDING MEAN
    # -----------------------------------------------------

    expanding_mean = (
        np.mean(hist)
        if len(hist) > 0
        else 0
    )


    # -----------------------------------------------------
    # DERIVED FEATURES
    # -----------------------------------------------------

    price_difference = (
        price - competitor_price
    )


    price_ratio = (
        price /
        (competitor_price + 1e-6)
    )


    discount_amount = (
        price *
        discount /
        100
    )


    holiday_weekend = (
        holiday_flag *
        is_weekend
    )


    # -----------------------------------------------------
    # REVENUE
    # -----------------------------------------------------

    revenue = (
        price *
        stock *
        (1 - discount / 100)
    )


    # -----------------------------------------------------
    # CREATE BASE ROW
    # -----------------------------------------------------

    row = {

        "Row_ID": 0,

        "Price": price,

        "Discount_Percentage":
            discount,

        "Revenue":
            revenue,

        "Promotion_Flag":
            promotion_flag,

        "Stock_Availability":
            stock,

        "Day_of_Week":
            day_of_week,

        "Month":
            month,

        "Quarter":
            quarter,

        "Holiday_Flag":
            holiday_flag,

        "Is_Weekend":
            is_weekend,

        "Local_Event_Flag":
            local_event_flag,

        "Competitor_Price":
            competitor_price,

        "Economic_Indicator":
            economic_indicator,

        "Marketing_Spend":
            marketing_spend,

        "Year":
            year,

        "Day":
            day,

        "Week_of_Year":
            week_of_year,

        "Is_Weekday":
            is_weekday,

        "Month_Sin":
            month_sin,

        "Month_Cos":
            month_cos,

        "DayOfWeek_Sin":
            dayofweek_sin,

        "DayOfWeek_Cos":
            dayofweek_cos,

        "Units_Sold_Lag_1":
            lag_1,

        "Units_Sold_Lag_7":
            lag_7,

        "Units_Sold_Lag_14":
            lag_14,

        "Units_Sold_Lag_30":
            lag_30,

        "Sales_Rolling_Mean_7":
            rolling_mean_7,

        "Sales_Rolling_Mean_14":
            rolling_mean_14,

        "Sales_Rolling_Mean_30":
            rolling_mean_30,

        "Sales_Rolling_Max_7":
            rolling_max_7,

        "Sales_Rolling_Min_7":
            rolling_min_7,

        "Sales_Rolling_Std_7":
            rolling_std_7,

        "Sales_Expanding_Mean":
            expanding_mean,

        "Time_Index":
            len(df) + 1,

        "Price_Difference":
            price_difference,

        "Price_Ratio":
            price_ratio,

        "Discount_Amount":
            discount_amount,

        "Holiday_Weekend":
            holiday_weekend
    }


    # =====================================================
    # ALTERNATIVE FEATURE NAMES
    # =====================================================

    # Reference/project variants

    row["Rolling_Mean_7"] = rolling_mean_7

    row["Rolling_Mean_14"] = rolling_mean_14

    row["Rolling_7_Max"] = rolling_max_7

    row["Rolling_7_Min"] = rolling_min_7

    row["Rolling_7_Std"] = rolling_std_7

    row["Product_Avg_Units_Sold"] = (
        np.mean(hist)
        if len(hist) > 0
        else 0
    )

    row["Store_Avg_Units_Sold"] = (
        np.mean(hist)
        if len(hist) > 0
        else 0
    )


    # =====================================================
    # PRODUCT / STORE ONE-HOT FEATURES
    # =====================================================

    row[
        f"Product_ID_{product_id}"
    ] = 1


    row[
        f"Product_Name_{product_name}"
    ] = 1


    row[
        f"Store_ID_{store_id}"
    ] = 1


    row[
        f"Store_Location_{store_location}"
    ] = 1


    # =====================================================
    # STATIC CATEGORICAL FEATURES
    # =====================================================

    # Take the latest matching row as template

    template = df[
        (df["Product_ID"] == product_id)
        &
        (df["Store_ID"] == store_id)
    ]


    if not template.empty:

        template_row = (
            template
            .sort_values("Date")
            .iloc[-1]
        )

        for col in FEATURE_COLUMNS:

            if (
                col not in row
                and col in template_row.index
            ):

                value = template_row[col]

                if pd.notna(value):

                    row[col] = value


    # =====================================================
    # FINAL FEATURE DATAFRAME
    # =====================================================

    final_row = {}

    for col in FEATURE_COLUMNS:

        if col in row:

            final_row[col] = row[col]

        else:

            final_row[col] = 0


    final_df = pd.DataFrame(
        [final_row]
    )


    # =====================================================
    # FEATURE ORDER
    # =====================================================

    final_df = final_df[
        FEATURE_COLUMNS
    ]


    # =====================================================
    # NUMERIC CONVERSION
    # =====================================================

    final_df = final_df.apply(
        pd.to_numeric,
        errors="coerce"
    )


    final_df = final_df.fillna(0)


    return final_df


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("---")


predict_button = st.button(
    "🔮 Predict Units Sold",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # =================================================
        # PRODUCT + STORE HISTORY
        # =================================================

        history_df = df[
            (df["Product_ID"] == selected_product_id)
            &
            (df["Store_ID"] == store)
        ].copy()


        history_df = history_df.sort_values(
            "Date"
        )


        if history_df.empty:

            st.error(
                "❌ No historical data found "
                "for selected Product and Store."
            )

            st.stop()


        sales_history = (
            history_df[
                "Units_Sold"
            ]
            .dropna()
            .astype(float)
            .tolist()
        )


        # =================================================
        # FUTURE DATES
        # =================================================

        current_date = pd.Timestamp(
            forecast_date
        )


        future_dates = pd.date_range(

            start=current_date,

            periods=forecast_days,

            freq="D"
        )


        predictions = []


        # =================================================
        # RECURSIVE FORECAST
        # =================================================

        for future_date in future_dates:

            X_future = build_future_row(

                date=future_date,

                product_id=selected_product_id,

                store_id=store,

                product_name=product,

                store_location=selected_store_location,

                history=sales_history
            )


            prediction = model.predict(
                X_future
            )[0]


            prediction = max(
                0,
                float(prediction)
            )


            prediction = int(
                round(prediction)
            )


            predictions.append({

                "Date":
                    future_date,

                "Predicted_Units_Sold":
                    prediction

            })


            # Add predicted value
            # to history for next day

            sales_history.append(
                prediction
            )


        # =================================================
        # FORECAST DATAFRAME
        # =================================================

        forecast_df = pd.DataFrame(
            predictions
        )


        # =================================================
        # PREDICTION RESULT
        # =================================================

        st.markdown("---")


        st.header(
            "🎯 Prediction Result"
        )


        st.success(
            "Prediction completed successfully!"
        )


        # =================================================
        # SINGLE DAY
        # =================================================

        if forecast_days == 1:

            predicted_units = int(
                forecast_df[
                    "Predicted_Units_Sold"
                ].iloc[0]
            )


            st.metric(

                label="📦 Predicted Units Sold",

                value=f"{predicted_units:,.0f}"

            )


        # =================================================
        # MULTI-DAY FORECAST SUMMARY
        # =================================================

        else:

            total_forecast = int(
                forecast_df[
                    "Predicted_Units_Sold"
                ].sum()
            )


            average_forecast = int(
                round(
                    forecast_df[
                        "Predicted_Units_Sold"
                    ].mean()
                )
            )


            highest_forecast = int(
                forecast_df[
                    "Predicted_Units_Sold"
                ].max()
            )


            lowest_forecast = int(
                forecast_df[
                    "Predicted_Units_Sold"
                ].min()
            )


            st.subheader(
                "📊 Forecast Summary"
            )


            c1, c2, c3, c4 = st.columns(4)


            with c1:

                st.metric(
                    "Total Forecast",
                    f"{total_forecast:,} Units"
                )


            with c2:

                st.metric(
                    "Average Daily Sales",
                    f"{average_forecast:,} Units"
                )


            with c3:

                st.metric(
                    "Highest Daily Sales",
                    f"{highest_forecast:,} Units"
                )


            with c4:

                st.metric(
                    "Lowest Daily Sales",
                    f"{lowest_forecast:,} Units"
                )


            # =================================================
            # FORECAST CHART
            # =================================================

            st.subheader(
                "📈 Sales Forecast"
            )


            chart_df = (
                forecast_df
                .set_index("Date")
            )


            st.line_chart(
                chart_df[
                    "Predicted_Units_Sold"
                ]
            )


            # =================================================
            # FORECAST TABLE
            # =================================================

            st.subheader(
                "📋 Forecast Details"
            )


            display_forecast = (
                forecast_df.copy()
            )


            display_forecast["Date"] = (
                display_forecast["Date"]
                .dt.strftime(
                    "%d-%m-%Y"
                )
            )


            display_forecast = (
                display_forecast.rename(
                    columns={

                        "Date":
                            "Forecast Date",

                        "Predicted_Units_Sold":
                            "Predicted Units Sold"

                    }
                )
            )


            st.dataframe(
                display_forecast,
                use_container_width=True,
                hide_index=True
            )


        # =================================================
        # INPUT SUMMARY
        # =================================================

        st.subheader(
            "📋 Input Summary"
        )


        summary = pd.DataFrame({

            "Input": [

                "Forecast Date",

                "Product",

                "Store",

                "Price",

                "Discount %",

                "Promotion",

                "Stock Availability",

                "Holiday",

                "Local Event",

                "Competitor Price",

                "Economic Indicator",

                "Marketing Spend",

                "Forecast Horizon"

            ],

            "Value": [

                str(forecast_date),

                product,

                store,

                price,

                discount,

                promotion,

                stock,

                holiday,

                local_event,

                competitor_price,

                economic_indicator,

                marketing_spend,

                forecast_horizon

            ]

        })


        st.dataframe(

            summary,

            use_container_width=True,

            hide_index=True

        )


        # =================================================
        # DOWNLOAD FORECAST
        # =================================================

        if forecast_days > 1:

            csv_data = (
                forecast_df
                .to_csv(index=False)
                .encode("utf-8")
            )


            st.download_button(

                label="⬇️ Download Forecast CSV",

                data=csv_data,

                file_name=(
                    "catboost_sales_forecast.csv"
                ),

                mime="text/csv",

                use_container_width=True

            )


    except FileNotFoundError:

        st.error(
            "❌ CatBoost model file not found!"
        )

        st.code(
            """
app.py
final_catboost_sales_model.pkl
Sales_Forcasting_Dataset_Corrected.xlsx
            """
        )


    except Exception as e:

        st.error(
            "❌ Prediction Error"
        )

        st.exception(e)