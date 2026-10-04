import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import hstack

model = pickle.load(open("big_mart_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Big Mart Sales Prediction")

item_weight = st.number_input("Item Weight", min_value=0.0)
item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
item_visibility = st.number_input("Item Visibility", min_value=0.0)
item_type = st.selectbox("Item Type", [
    "Baking Goods",
    "Breads",
    "Breakfast",
    "Canned",
    "Dairy",
    "Frozen Foods",
    "Fruits and Vegetables",
    "Hard Drinks",
    "Health and Hygiene",
    "Household",
    "Meat",
    "Others",
    "Seafood",
    "Snack Foods",
    "Soft Drinks",
    "Starchy Foods"
])
item_mrp = st.number_input("Item MRP", min_value=0.0)
outlet_identifier = st.selectbox("Outlet Identifier", [
    "OUT010",
    "OUT013",
    "OUT017",
    "OUT018",
    "OUT019",
    "OUT027",
    "OUT035",
    "OUT045",
    "OUT046",
    "OUT049"
])
outlet_establishment_year = st.number_input(
    "Outlet Establishment Year",
    min_value=1980,
    max_value=2026,
    value=1998
)
outlet_size = st.selectbox(
    "Outlet Size",
    ["Small", "Medium", "High"]
)
outlet_location_type = st.selectbox(
    "Outlet Location Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)
outlet_type = st.selectbox(
    "Outlet Type",
    ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
)

if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "Item_Weight": [item_weight],
        "Item_Fat_Content": [item_fat_content],
        "Item_Visibility": [item_visibility],
        "Item_Type": [item_type],
        "Item_MRP": [item_mrp],
        "Outlet_Identifier": [outlet_identifier],
        "Outlet_Establishment_Year": [outlet_establishment_year],
        "Outlet_Size": [outlet_size],
        "Outlet_Location_Type": [outlet_location_type],
        "Outlet_Type": [outlet_type]
    })

    categorical_columns = input_data.select_dtypes(
        include="object"
    ).columns

    numerical_columns = input_data.select_dtypes(
        exclude="object"
    ).columns

    encoded_data = encoder.transform(
        input_data[categorical_columns]
    )

    numerical_data = scaler.transform(
        input_data[numerical_columns]
    )

    final_data = hstack([
        encoded_data,
        numerical_data
    ])

    prediction = model.predict(final_data)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")

