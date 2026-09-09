import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==============================================================
# PAGE CONFIG
# ==============================================================


st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# ==============================================================
# FILE PATHS
# ==============================================================

MODEL_FILE = "final_xgboost_sales_model.pkl"

DATA_FILE = "Sales_Forcasting_Dataset_Corrected.xlsx"

FEATURE_FILE = "xgboost_feature_columns.pkl"


# ==============================================================
# LOAD MODEL
# ==============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    return model


# ==============================================================
# LOAD FEATURE COLUMNS
# ==============================================================

@st.cache_resource
def load_feature_columns():

    feature_columns = joblib.load(
        FEATURE_FILE
    )

    if isinstance(
        feature_columns,
        np.ndarray
    ):
        feature_columns = (
            feature_columns.tolist()
        )

    elif isinstance(
        feature_columns,
        tuple
    ):
        feature_columns = list(
            feature_columns
        )

    elif isinstance(
        feature_columns,
        pd.DataFrame
    ):
        feature_columns = (
            feature_columns.columns.tolist()
        )

    else:
        feature_columns = list(
            feature_columns
        )

    return feature_columns


# ==============================================================
# LOAD DATASET
# ==============================================================

@st.cache_data
def load_dataset():

    data = pd.read_excel(
        DATA_FILE
    )

    # ----------------------------------------------------------
    # Date
    # ----------------------------------------------------------

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    # ----------------------------------------------------------
    # Holiday
    # ----------------------------------------------------------

    if "Holiday_Name" in data.columns:

        data["Holiday_Name"] = (
            data["Holiday_Name"]
            .fillna(
                "No Government Holiday"
            )
        )

    # ----------------------------------------------------------
    # Sort
    # ----------------------------------------------------------

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return data


# ==============================================================
# LOAD ALL
# ==============================================================

try:

    model = load_model()

    FEATURE_COLUMNS = (
        load_feature_columns()
    )

    df = load_dataset()

except FileNotFoundError as e:

    st.error(
        "❌ Required file not found!"
    )

    st.info(
        """
Make sure these files are in the same folder:

new_app.py
final_xgboost_sales_model.pkl
xgboost_feature_columns.pkl
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


# ==============================================================
# STATIC / ONE-HOT FEATURES
# ==============================================================

STATIC_PREFIXES = [

    "Product_ID_",

    "Product_Name_",

    "Category_",

    "Store_ID_",

    "Store_Location_",

    "Holiday_Name_",

    "Season_",

    "Weather_",

    "Sales_Channel_",

    "Customer_Segment_"

]


# ==============================================================
# PRODUCT OPTIONS
# ==============================================================

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


product_options = []

for _, row in product_data.iterrows():

    product_id = row[
        "Product_ID"
    ]

    product_name = row[
        "Product_Name"
    ]

    product_options.append(
        (
            str(product_id),
            str(product_name)
        )
    )


# ==============================================================
# STORE OPTIONS
# ==============================================================

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


store_options = []

for _, row in store_data.iterrows():

    store_id = row[
        "Store_ID"
    ]

    location = row[
        "Store_Location"
    ]

    store_options.append(
        (
            str(store_id),
            str(location)
        )
    )


# ==============================================================
# REGION OPTIONS
# ==============================================================

region_options = sorted(
    df[
        "Store_Location"
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ==============================================================
# HELPER
# ==============================================================

def get_latest_value(
    data,
    column,
    default=0
):
    """
    Safely get the latest non-null value from either:
    - a pandas DataFrame
    - a pandas Series (for example, df.iloc[-1])
    - a dictionary-like object

    This prevents:
        AttributeError: 'Series' object has no attribute 'columns'
    """

    # DataFrame
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            return default

        values = data[column].dropna()

        if values.empty:
            return default

        return values.iloc[-1]

    # Series (e.g. template_row = df.iloc[-1])
    if isinstance(data, pd.Series):
        if column not in data.index:
            return default

        value = data[column]

        if pd.isna(value):
            return default

        return value

    # Dictionary / mapping-like object
    if isinstance(data, dict):
        if column not in data:
            return default

        value = data[column]

        if pd.isna(value):
            return default

        return value

    # Generic fallback
    try:
        value = data[column]

        if pd.isna(value):
            return default

        return value

    except (KeyError, TypeError, IndexError):
        return default


# ==============================================================
# BUILD FUTURE ROW
# ==============================================================

def build_future_row(
    date,
    product_id,
    store_id,
    product_name,
    store_location,
    history,
    template_row,
    user_inputs=None
):

    date = pd.Timestamp(
        date
    )


    # ==========================================================
    # CALENDAR FEATURES
    # ==========================================================

    year = date.year

    month = date.month

    day = date.day

    day_of_week = (
        date.dayofweek
    )

    quarter = (
        date.quarter
    )

    week_of_year = int(
        date.isocalendar().week
    )

    day_of_year = (
        date.dayofyear
    )


    # ==========================================================
    # WEEK FEATURES
    # ==========================================================

    is_weekend = int(
        day_of_week >= 5
    )

    is_weekday = int(
        day_of_week < 5
    )

    is_month_start = int(
        date.is_month_start
    )

    is_month_end = int(
        date.is_month_end
    )


    # ==========================================================
    # CYCLICAL FEATURES
    # ==========================================================

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


    # ==========================================================
    # HISTORY
    # ==========================================================

    hist = list(
        history
    )

    hist = [
        float(x)
        for x in hist
        if pd.notna(x)
    ]


    # ==========================================================
    # LAG FEATURES
    # ==========================================================

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


    # ==========================================================
    # ROLLING FEATURES
    # ==========================================================

    last7 = hist[-7:]

    last14 = hist[-14:]

    last30 = hist[-30:]


    if len(last7) > 0:

        rolling_mean_7 = (
            np.mean(last7)
        )

        rolling_max_7 = (
            np.max(last7)
        )

        rolling_min_7 = (
            np.min(last7)
        )

    else:

        rolling_mean_7 = 0

        rolling_max_7 = 0

        rolling_min_7 = 0


    if len(last14) > 0:

        rolling_mean_14 = (
            np.mean(last14)
        )

    else:

        rolling_mean_14 = 0


    if len(last30) > 0:

        rolling_mean_30 = (
            np.mean(last30)
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


    # ==========================================================
    # EXPANDING MEAN
    # ==========================================================

    expanding_mean = (

        np.mean(hist)
        if len(hist) > 0
        else 0

    )


    # ==========================================================
    # INPUT VALUES
    # ==========================================================

    if user_inputs is not None:

        price = user_inputs[
            "Price"
        ]

        discount = user_inputs[
            "Discount_Percentage"
        ]

        promotion_flag = user_inputs[
            "Promotion_Flag"
        ]

        stock = user_inputs[
            "Stock_Availability"
        ]

        holiday_flag = user_inputs[
            "Holiday_Flag"
        ]

        local_event_flag = user_inputs[
            "Local_Event_Flag"
        ]

        competitor_price = user_inputs[
            "Competitor_Price"
        ]

        economic_indicator = user_inputs[
            "Economic_Indicator"
        ]

        marketing_spend = user_inputs[
            "Marketing_Spend"
        ]

    else:

        # ------------------------------------------------------
        # STORE FORECAST
        # Automatically use latest historical values
        # ------------------------------------------------------

        price = get_latest_value(
            template_row,
            "Price",
            0
        )

        discount = get_latest_value(
            template_row,
            "Discount_Percentage",
            0
        )

        promotion_flag = get_latest_value(
            template_row,
            "Promotion_Flag",
            0
        )

        stock = get_latest_value(
            template_row,
            "Stock_Availability",
            0
        )

        holiday_flag = get_latest_value(
            template_row,
            "Holiday_Flag",
            0
        )

        local_event_flag = get_latest_value(
            template_row,
            "Local_Event_Flag",
            0
        )

        competitor_price = get_latest_value(
            template_row,
            "Competitor_Price",
            0
        )

        economic_indicator = get_latest_value(
            template_row,
            "Economic_Indicator",
            0
        )

        marketing_spend = get_latest_value(
            template_row,
            "Marketing_Spend",
            0
        )


    # ==========================================================
    # DERIVED FEATURES
    # ==========================================================

    price_difference = (
        price
        - competitor_price
    )

    price_ratio = (
        price
        / (competitor_price + 1e-6)
    )

    discount_amount = (
        price
        * discount
        / 100
    )

    holiday_weekend = (
        holiday_flag
        * is_weekend
    )


    # ==========================================================
    # REVENUE
    # ==========================================================

    revenue = (
        price
        * stock
        * (1 - discount / 100)
    )


    # ==========================================================
    # BASE ROW
    # ==========================================================

    row = {

        "Row_ID":
            get_latest_value(
                template_row,
                "Row_ID",
                0
            ),

        "Price":
            price,

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

        "Day_of_Year":
            day_of_year,

        "Is_Weekday":
            is_weekday,

        "Is_Month_Start":
            is_month_start,

        "Is_Month_End":
            is_month_end,

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
            len(df) + len(hist) + 1,

        "Price_Difference":
            price_difference,

        "Price_Ratio":
            price_ratio,

        "Discount_Amount":
            discount_amount,

        "Holiday_Weekend":
            holiday_weekend
    }


    # ==========================================================
    # ALTERNATIVE FEATURE NAMES
    # ==========================================================

    row[
        "Rolling_Mean_7"
    ] = rolling_mean_7

    row[
        "Rolling_Mean_14"
    ] = rolling_mean_14

    row[
        "Rolling_7_Max"
    ] = rolling_max_7

    row[
        "Rolling_7_Min"
    ] = rolling_min_7

    row[
        "Rolling_7_Std"
    ] = rolling_std_7

    row[
        "Product_Avg_Units_Sold"
    ] = (

        np.mean(hist)
        if len(hist) > 0
        else 0

    )

    row[
        "Store_Avg_Units_Sold"
    ] = (

        np.mean(hist)
        if len(hist) > 0
        else 0

    )


    # ==========================================================
    # PRODUCT / STORE ONE-HOT
    # ==========================================================

    row[
        f"Product_ID_{product_id}"
    ] = 1

    row[
        f"Store_ID_{store_id}"
    ] = 1

    row[
        f"Store_Location_{store_location}"
    ] = 1

    row[
        f"Product_Name_{product_name}"
    ] = 1


    # ==========================================================
    # STATIC CATEGORICAL FEATURES
    # ==========================================================

    if template_row is not None:

        for col in FEATURE_COLUMNS:

            if (
                col not in row
                and col in template_row.index
            ):

                value = (
                    template_row[col]
                )

                if pd.notna(value):

                    row[col] = value


    # ==========================================================
    # FINAL ROW
    # ==========================================================

    final_row = {}

    for col in FEATURE_COLUMNS:

        if col in row:

            final_row[col] = (
                row[col]
            )

        else:

            final_row[col] = 0


    final_df = pd.DataFrame(
        [final_row]
    )


    # ==========================================================
    # FEATURE ORDER
    # ==========================================================

    final_df = final_df[
        FEATURE_COLUMNS
    ]


    # ==========================================================
    # NUMERIC
    # ==========================================================

    final_df = final_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    final_df = final_df.fillna(
        0
    )


    return final_df


# ==============================================================
# PRODUCT FORECAST
# ==============================================================

def forecast_future_sales(
    product_id,
    store_id,
    current_date,
    forecast_days,
    user_inputs
):

    product_history_df = df[
        df["Product_ID"].astype(str)
        == str(product_id)
    ].copy()


    store_history_df = df[
        df["Store_ID"].astype(str)
        == str(store_id)
    ].copy()


    combo_history_df = df[
        (
            df["Product_ID"].astype(str)
            == str(product_id)
        )
        &
        (
            df["Store_ID"].astype(str)
            == str(store_id)
        )
    ].copy()


    combo_history_df = (
        combo_history_df
        .sort_values("Date")
    )


    if combo_history_df.empty:

        raise ValueError(
            f"No historical data found for "
            f"Product {product_id} "
            f"and Store {store_id}."
        )


    template_row = (
        combo_history_df.iloc[-1]
    )


    # ----------------------------------------------------------
    # PRODUCT HISTORY
    # ----------------------------------------------------------

    product_history = (
        product_history_df
        .sort_values("Date")
        ["Units_Sold"]
        .dropna()
        .astype(float)
        .tolist()
    )


    # ----------------------------------------------------------
    # STORE HISTORY
    # ----------------------------------------------------------

    store_history = (
        store_history_df
        .sort_values("Date")
        ["Units_Sold"]
        .dropna()
        .astype(float)
        .tolist()
    )


    # ----------------------------------------------------------
    # FUTURE DATES
    # ----------------------------------------------------------

    first_forecast_date = (
        pd.Timestamp(current_date)
        + pd.Timedelta(days=1)
    )


    future_dates = pd.date_range(
        start=first_forecast_date,
        periods=forecast_days,
        freq="D"
    )


    predictions = []


    # ==========================================================
    # RECURSIVE FORECAST
    # ==========================================================

    for date in future_dates:

        row = build_future_row(

            date=date,

            product_id=str(
                product_id
            ),

            store_id=str(
                store_id
            ),

            product_name=str(
                template_row[
                    "Product_Name"
                ]
            ),

            store_location=str(
                template_row[
                    "Store_Location"
                ]
            ),

            history=product_history,

            template_row=template_row,

            user_inputs=user_inputs

        )


        prediction = model.predict(
            row
        )[0]


        prediction = max(
            0,
            int(
                round(
                    float(
                        prediction
                    )
                )
            )
        )


        predictions.append({

            "Date":
                date,

            "Predicted_Units_Sold":
                prediction

        })


        # ------------------------------------------------------
        # Add prediction for recursive forecasting
        # ------------------------------------------------------

        product_history.append(
            prediction
        )

        store_history.append(
            prediction
        )


    return (
        pd.DataFrame(predictions),
        combo_history_df
    )


# ==============================================================
# REGION PRODUCT-WISE FORECAST
# ==============================================================

def forecast_region_product_sales(
    region,
    current_date,
    forecast_days
):

    region_df = df[
        df["Store_Location"].astype(str)
        == str(region)
    ].copy()


    if region_df.empty:

        raise ValueError(
            f"No historical data found "
            f"for Region {region}."
        )


    future_dates = pd.date_range(

        start=(
            pd.Timestamp(current_date)
            + pd.Timedelta(days=1)
        ),

        periods=forecast_days,

        freq="D"

    )


    predictions = []


    # ----------------------------------------------------------
    # PRODUCTS IN REGION
    # ----------------------------------------------------------

    region_products = sorted(

        region_df[
            "Product_ID"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()

    )


    for product_id in region_products:

        product_region_df = (
            region_df[
                region_df[
                    "Product_ID"
                ].astype(str)
                == str(product_id)
            ]
            .sort_values("Date")
            .copy()
        )


        if product_region_df.empty:

            continue


        # ------------------------------------------------------
        # DAILY PRODUCT SALES
        # ------------------------------------------------------

        daily_product_sales = (

            product_region_df
            .groupby("Date")
            ["Units_Sold"]
            .sum()
            .sort_index()

        )


        if daily_product_sales.empty:

            continue


        # ------------------------------------------------------
        # PRODUCT NAME
        # ------------------------------------------------------

        product_name = str(
            product_region_df.iloc[-1][
                "Product_Name"
            ]
        )


        # ------------------------------------------------------
        # RECENT HISTORY
        # ------------------------------------------------------

        recent_window = min(
            30,
            len(
                daily_product_sales
            )
        )


        recent_sales = (
            daily_product_sales
            .tail(recent_window)
            .astype(float)
        )


        overall_mean = (

            recent_sales.mean()
            if len(recent_sales) > 0
            else 0

        )


        # ------------------------------------------------------
        # DAY OF WEEK
        # ------------------------------------------------------

        dow_means = (

            recent_sales
            .groupby(
                recent_sales.index.dayofweek
            )
            .mean()

        )


        # ------------------------------------------------------
        # FORECAST
        # ------------------------------------------------------

        for date in future_dates:

            dow = date.dayofweek


            if dow in dow_means.index:

                prediction = (
                    dow_means.loc[dow]
                )

            else:

                prediction = (
                    overall_mean
                )


            prediction = max(

                0,

                int(
                    round(
                        float(
                            prediction
                        )
                    )
                )

            )


            predictions.append({

                "Date":
                    date,

                "Product":
                    product_id,

                "Product_Name":
                    product_name,

                "Predicted_Units_Sold":
                    prediction

            })


    if not predictions:

        raise ValueError(
            f"No product sales history "
            f"found for Region {region}."
        )


    return pd.DataFrame(
        predictions
    )


# ==============================================================
# STORE PRODUCT-WISE FORECAST
# ==============================================================

def forecast_store_product_sales(
    store_id,
    current_date,
    forecast_days
):

    store_df = df[
        df["Store_ID"].astype(str)
        == str(store_id)
    ].copy()


    if store_df.empty:

        raise ValueError(
            f"No historical data found "
            f"for Store {store_id}."
        )


    future_dates = pd.date_range(

        start=(
            pd.Timestamp(current_date)
            + pd.Timedelta(days=1)
        ),

        periods=forecast_days,

        freq="D"

    )


    predictions = []


    # ----------------------------------------------------------
    # PRODUCTS AVAILABLE IN STORE
    # ----------------------------------------------------------

    store_products = sorted(

        store_df[
            "Product_ID"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()

    )


    for product_id in store_products:

        product_store_df = (

            store_df[
                store_df[
                    "Product_ID"
                ].astype(str)
                == str(product_id)
            ]

            .sort_values("Date")

            .copy()

        )


        if product_store_df.empty:

            continue


        template_row = (
            product_store_df.iloc[-1]
        )


        # ------------------------------------------------------
        # PRODUCT HISTORY
        # ------------------------------------------------------

        product_history = (

            product_store_df[
                "Units_Sold"
            ]

            .dropna()

            .astype(float)

            .tolist()

        )


        # ------------------------------------------------------
        # STORE HISTORY
        # ------------------------------------------------------

        store_history = (

            store_df
            .sort_values("Date")
            ["Units_Sold"]
            .dropna()
            .astype(float)
            .tolist()

        )


        # ------------------------------------------------------
        # PRODUCT NAME
        # ------------------------------------------------------

        product_name = str(
            template_row[
                "Product_Name"
            ]
        )


        # ------------------------------------------------------
        # STORE LOCATION
        # ------------------------------------------------------

        store_location = str(
            template_row[
                "Store_Location"
            ]
        )


        # ------------------------------------------------------
        # RECURSIVE PRODUCT FORECAST
        # ------------------------------------------------------

        for date in future_dates:

            X_future = build_future_row(

                date=date,

                product_id=str(
                    product_id
                ),

                store_id=str(
                    store_id
                ),

                product_name=product_name,

                store_location=store_location,

                history=product_history,

                template_row=template_row,

                user_inputs=None

            )


            prediction = model.predict(
                X_future
            )[0]


            prediction = max(

                0,

                int(
                    round(
                        float(
                            prediction
                        )
                    )
                )

            )


            predictions.append({

                "Date":
                    date,

                "Product":
                    product_id,

                "Product_Name":
                    product_name,

                "Predicted_Units_Sold":
                    prediction

            })


            # --------------------------------------------------
            # Recursive update
            # --------------------------------------------------

            product_history.append(
                prediction
            )

            store_history.append(
                prediction
            )


    if not predictions:

        raise ValueError(
            f"No product forecast "
            f"available for Store {store_id}."
        )


    return pd.DataFrame(
        predictions
    )


# ==============================================================
# APP TITLE
# ==============================================================

st.title(
    "📈 Sales Forecasting - XGBoost Model"
)

st.caption(
    "XGBoost-powered future sales forecasting "
    "with Product, Store and Region-wise analysis"
)


# ==============================================================
# SIDEBAR
# ==============================================================

with st.sidebar:

    st.header(
        "🔮 Sales Forecasting"
    )


    # ==========================================================
    # PRODUCT BASED FORECAST
    # ==========================================================

    st.subheader(
        "📦 Product-Based Forecast"
    )


    product_labels = [

        f"{pid} - {pname}"

        for pid, pname
        in product_options

    ]


    selected_product = st.selectbox(

        "📦 Product",

        product_labels,

        key="product_select"

    )


    selected_product_index = (
        product_labels.index(
            selected_product
        )
    )


    product_id = (
        product_options[
            selected_product_index
        ][0]
    )


    # ----------------------------------------------------------
    # STORE
    # ----------------------------------------------------------

    store_labels = [

        f"{sid} - {location}"

        for sid, location
        in store_options

    ]


    selected_store = st.selectbox(

        "🏪 Store",

        store_labels,

        key="store_select"

    )


    selected_store_index = (
        store_labels.index(
            selected_store
        )
    )


    store_id = (
        store_options[
            selected_store_index
        ][0]
    )


    # ----------------------------------------------------------
    # CURRENT DATE
    # ----------------------------------------------------------

    product_current_date = st.date_input(

        "📅 Current Date",

        value=(
            pd.Timestamp
            .today()
            .date()
        ),

        key="product_current_date"

    )


    product_current_date = (
        pd.Timestamp(
            product_current_date
        )
    )


    # ----------------------------------------------------------
    # BUSINESS INPUTS
    # ----------------------------------------------------------

    price = st.number_input(

        "💰 Price",

        min_value=0.0,

        value=1000.0,

        step=1.0,

        key="product_price"

    )


    discount = st.number_input(

        "🏷️ Discount %",

        min_value=0.0,

        max_value=100.0,

        value=0.0,

        step=1.0,

        key="product_discount"

    )


    promotion = st.selectbox(

        "📢 Promotion",

        [
            "Yes",
            "No"
        ],

        key="product_promotion"

    )


    promotion_flag = (

        1
        if promotion == "Yes"
        else 0

    )


    stock = st.number_input(

        "📦 Stock Availability",

        min_value=0.0,

        value=100.0,

        step=1.0,

        key="product_stock"

    )


    holiday = st.selectbox(

        "🎉 Holiday",

        [
            "Yes",
            "No"
        ],

        key="product_holiday"

    )


    holiday_flag = (

        1
        if holiday == "Yes"
        else 0

    )


    local_event = st.selectbox(

        "📍 Local Event",

        [
            "Yes",
            "No"
        ],

        key="product_local_event"

    )


    local_event_flag = (

        1
        if local_event == "Yes"
        else 0

    )


    competitor_price = st.number_input(

        "💰 Competitor Price",

        min_value=0.0,

        value=1000.0,

        step=1.0,

        key="product_competitor_price"

    )


    economic_indicator = st.number_input(

        "📊 Economic Indicator",

        value=100.0,

        step=0.1,

        key="product_economic_indicator"

    )


    marketing_spend = st.number_input(

        "📣 Marketing Spend",

        min_value=0.0,

        value=5000.0,

        step=100.0,

        key="product_marketing_spend"

    )


    # ----------------------------------------------------------
    # PRODUCT HORIZON
    # ----------------------------------------------------------

    product_forecast_horizon = st.selectbox(

        "🔮 Forecast Horizon",

        [
            "1 Day",
            "7 Days",
            "30 Days",
            "90 Days",
            "1 Year",
            "2 Years"
        ],

        key="product_forecast_horizon"

    )


    horizon_map = {

        "1 Day": 1,

        "7 Days": 7,

        "30 Days": 30,

        "90 Days": 90,

        "1 Year": 365,

        "2 Years": 730

    }


    product_forecast_days = (
        horizon_map[
            product_forecast_horizon
        ]
    )


    product_run_button = st.button(

        "🚀 Generate Product Forecast",

        type="primary",

        use_container_width=True,

        key="product_run_button"

    )
    # ==========================================================
    # STORE BASED FORECAST
    # ==========================================================

    st.divider()

    st.subheader(
        "🏪 Store-Based Forecast"
    )


    store_forecast_labels = [

        f"{sid} - {location}"

        for sid, location
        in store_options

    ]


    selected_store_forecast = (
        st.selectbox(

            "🏪 Store",

            store_forecast_labels,

            key="store_forecast_select"

        )
    )


    selected_store_forecast_index = (
        store_forecast_labels.index(
            selected_store_forecast
        )
    )


    store_forecast_id = (
        store_options[
            selected_store_forecast_index
        ][0]
    )


    # ----------------------------------------------------------
    # STORE DATE
    # ----------------------------------------------------------

    store_current_date = st.date_input(

        "📅 Current Date",

        value=(
            pd.Timestamp
            .today()
            .date()
        ),

        key="store_current_date"

    )


    store_current_date = (
        pd.Timestamp(
            store_current_date
        )
    )


    # ----------------------------------------------------------
    # STORE HORIZON
    # ----------------------------------------------------------

    store_forecast_horizon = st.selectbox(

        "🔮 Forecast Horizon",

        [
            "1 Day",
            "7 Days",
            "30 Days",
            "90 Days",
            "1 Year",
            "2 Years"
        ],

        key="store_forecast_horizon"

    )


    store_forecast_days = (
        horizon_map[
            store_forecast_horizon
        ]
    )


    store_run_button = st.button(

        "🏪 Generate Store Forecast",

        type="primary",

        use_container_width=True,

        key="store_run_button"

    )


# ==============================================================
# PRODUCT FORECAST OUTPUT
# ==============================================================

if product_run_button:

    user_inputs = {

        "Price":
            price,

        "Discount_Percentage":
            discount,

        "Promotion_Flag":
            promotion_flag,

        "Stock_Availability":
            stock,

        "Holiday_Flag":
            holiday_flag,

        "Local_Event_Flag":
            local_event_flag,

        "Competitor_Price":
            competitor_price,

        "Economic_Indicator":
            economic_indicator,

        "Marketing_Spend":
            marketing_spend

    }


    try:

        forecast_df, combo_history_df = (
            forecast_future_sales(

                product_id=
                    product_id,

                store_id=
                    store_id,

                current_date=
                    product_current_date,

                forecast_days=
                    product_forecast_days,

                user_inputs=
                    user_inputs

            )
        )


        # ======================================================
        # METRICS
        # ======================================================

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


        maximum_forecast = int(

            forecast_df[
                "Predicted_Units_Sold"
            ].max()

        )


        minimum_forecast = int(

            forecast_df[
                "Predicted_Units_Sold"
            ].min()

        )


        st.success(

            f"Product forecast generated successfully "
            f"for {selected_product} at {selected_store}."

        )


        st.subheader(

            "📊 Product / Store Forecast Summary"

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

                f"{maximum_forecast:,} Units"

            )


        with c4:

            st.metric(

                "Lowest Daily Sales",

                f"{minimum_forecast:,} Units"

            )


        # ======================================================
        # PRODUCT / STORE CHART
        # ======================================================

        st.subheader(

            "📈 Product / Store Sales Forecast"

        )


        recent_actual = (

            combo_history_df[
                [
                    "Date",
                    "Units_Sold"
                ]
            ]

            .tail(30)

            .rename(

                columns={
                    "Units_Sold":
                        "Units"
                }

            )

        )


        recent_actual["Type"] = (
            "Actual"
        )


        future_plot = (

            forecast_df

            .rename(

                columns={
                    "Predicted_Units_Sold":
                        "Units"
                }

            )

        )


        future_plot["Type"] = (
            "Forecast"
        )


        combined = pd.concat(

            [
                recent_actual,
                future_plot
            ],

            ignore_index=True

        )


        # ----------------------------------------------------------
        # IMPORTANT:
        # A Date + Type combination can occur more than once when
        # the historical dataset contains duplicate dates.
        # DataFrame.pivot() requires unique index/column pairs and
        # therefore raises:
        # "Index contains duplicate entries, cannot reshape"
        #
        # pivot_table() safely aggregates duplicate Date + Type rows.
        # ----------------------------------------------------------
        chart_data = (
            combined
            .groupby(
                ["Date", "Type"],
                as_index=False
            )["Units"]
            .sum()
            .pivot_table(
                index="Date",
                columns="Type",
                values="Units",
                aggfunc="sum"
            )
            .sort_index()
        )


        st.line_chart(
            chart_data
        )


        # ======================================================
        # FUTURE TABLE
        # ======================================================

        st.subheader(

            "📋 Future Product Sales Forecast"

        )


        display_forecast = (
            forecast_df.copy()
        )


        display_forecast["Date"] = (

            display_forecast[
                "Date"
            ]

            .dt.strftime(
                "%d-%m-%Y"
            )

        )


        display_forecast[
            "Predicted_Units_Sold"
        ] = (

            display_forecast[
                "Predicted_Units_Sold"
            ]

            .astype(int)

        )


        display_forecast = (

            display_forecast

            .rename(

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


        # ======================================================
        # DOWNLOAD
        # ======================================================

        csv = (

            display_forecast

            .to_csv(
                index=False
            )

            .encode("utf-8")

        )


        st.download_button(

            label=
                "⬇️ Download Product Forecast CSV",

            data=csv,

            file_name=(
                f"forecast_"
                f"{product_id}_"
                f"{store_id}.csv"
            ),

            mime="text/csv",

            use_container_width=True

        )


    except Exception as e:

        st.error(

            f"❌ Product forecast generation failed: {e}"

        )

        st.exception(e)
# ==============================================================
# STORE FORECAST OUTPUT
# ==============================================================

if store_run_button:

    try:

        store_forecast_df = (
            forecast_store_product_sales(

                store_id=
                    store_forecast_id,

                current_date=
                    store_current_date,

                forecast_days=
                    store_forecast_days

            )
        )


        # ======================================================
        # STORE DAILY TOTAL
        # ======================================================

        store_daily_total = (

            store_forecast_df

            .groupby("Date")

            [
                "Predicted_Units_Sold"
            ]

            .sum()

            .sort_index()

        )


        total_store_forecast = int(

            store_daily_total.sum()

        )


        average_store_forecast = int(

            round(
                store_daily_total.mean()
            )

        )


        max_store_forecast = int(

            store_daily_total.max()

        )


        min_store_forecast = int(

            store_daily_total.min()

        )


        st.success(

            f"Store forecast generated successfully "
            f"for {selected_store_forecast}."

        )


        st.subheader(

            f"🏪 {selected_store_forecast} - "
            f"Product-wise Units Sold Forecast"

        )


        # ======================================================
        # STORE METRICS
        # ======================================================

        s1, s2, s3, s4 = (
            st.columns(4)
        )


        with s1:

            st.metric(

                "Total Store Forecast",

                f"{total_store_forecast:,} Units"

            )


        with s2:

            st.metric(

                "Average Daily Total",

                f"{average_store_forecast:,} Units"

            )


        with s3:

            st.metric(

                "Highest Daily Total",

                f"{max_store_forecast:,} Units"

            )


        with s4:

            st.metric(

                "Lowest Daily Total",

                f"{min_store_forecast:,} Units"

            )


        # ======================================================
        # PRODUCT SUMMARY
        # ======================================================

        product_summary = (

            store_forecast_df

            .groupby(
                [
                    "Product",
                    "Product_Name"
                ]
            )

            [
                "Predicted_Units_Sold"
            ]

            .sum()

            .reset_index()

        )


        product_summary = (

            product_summary

            .rename(

                columns={

                    "Product":
                        "Product",

                    "Product_Name":
                        "Product Name",

                    "Predicted_Units_Sold":
                        "Predicted Units Sold"

                }

            )

        )


        product_summary = (

            product_summary

            .sort_values(

                "Predicted Units Sold",

                ascending=False

            )

            .reset_index(drop=True)

        )


        # ======================================================
        # PRODUCT TABLE
        # ======================================================

        st.subheader(
            "📋 Product-wise Store Forecast"
        )


        st.dataframe(

            product_summary,

            use_container_width=True,

            hide_index=True

        )


        # ======================================================
        # STORE DAILY TOTAL
        # ======================================================

        daily_store_table = (

            store_daily_total

            .reset_index()

            .rename(

                columns={

                    "Date":
                        "Forecast Date",

                    "Predicted_Units_Sold":
                        "Total Predicted Units"

                }

            )

        )


        daily_store_table[
            "Forecast Date"
        ] = (

            daily_store_table[
                "Forecast Date"
            ]

            .dt.strftime(
                "%d-%m-%Y"
            )

        )


        st.subheader(
            "📊 Daily Store Total"
        )


        st.dataframe(

            daily_store_table,

            use_container_width=True,

            hide_index=True

        )


        # ======================================================
        # STORE TOTAL CHART
        # ======================================================

        st.subheader(

            f"📈 {selected_store_forecast} - "
            f"Daily Total Forecast"

        )


        st.line_chart(

            store_daily_total.rename(
                "Predicted Units Sold"
            )

        )


        # ======================================================
        # PRODUCT-WISE CHART
        # ======================================================

        st.subheader(

            f"📈 {selected_store_forecast} - "
            f"Product-wise Forecast"

        )


        # ----------------------------------------------------------
        # IMPORTANT:
        # Use pivot_table instead of pivot because duplicate
        # Date + Product combinations can exist in the forecast
        # result. pivot_table safely aggregates them.
        # ----------------------------------------------------------
        store_product_chart = (
            store_forecast_df
            .pivot_table(
                index="Date",
                columns="Product",
                values="Predicted_Units_Sold",
                aggfunc="sum"
            )
            .sort_index()
        )


        st.line_chart(
            store_product_chart
        )


        # ======================================================
        # DETAILED TABLE
        # ======================================================

        st.subheader(
            "📋 Daily Product Forecast"
        )


        display_store = (
            store_forecast_df.copy()
        )


        display_store["Date"] = (

            display_store[
                "Date"
            ]

            .dt.strftime(
                "%d-%m-%Y"
            )

        )


        display_store[
            "Predicted_Units_Sold"
        ] = (

            display_store[
                "Predicted_Units_Sold"
            ]

            .astype(int)

        )


        display_store = (

            display_store

            .rename(

                columns={

                    "Date":
                        "Forecast Date",

                    "Product":
                        "Product",

                    "Product_Name":
                        "Product Name",

                    "Predicted_Units_Sold":
                        "Predicted Units Sold"

                }

            )

        )


        st.dataframe(

            display_store,

            use_container_width=True,

            hide_index=True

        )


        # ======================================================
        # DOWNLOAD
        # ======================================================

        store_csv = (

            display_store

            .to_csv(
                index=False
            )

            .encode("utf-8")

        )


        st.download_button(

            label=
                "⬇️ Download Store Product Forecast CSV",

            data=store_csv,

            file_name=(

                f"store_product_forecast_"
                f"{store_forecast_id}.csv"

            ),

            mime="text/csv",

            use_container_width=True

        )


    except Exception as e:

        st.error(

            f"❌ Store forecast failed: {e}"

        )

        st.exception(e)


# ==============================================================
# INITIAL MESSAGE
# ==============================================================

if (

    not product_run_button


    and not store_run_button

):

    st.info(

        "Product Forecast: select Product + Store and provide "
        "business inputs. Region Forecast: select only Region, "
        "Current Date and Forecast Horizon. Store Forecast: "
        "select Store, Current Date and Forecast Horizon."

    )