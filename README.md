# Big Mart Sales Prediction

A machine learning project that predicts the sales of products in Big Mart outlets using product and outlet-related features.
https://martsalesprediction-mdswcksktzykyf9k6tpja5.streamlit.app/
## Project Overview

The goal of this project is to predict `Item_Outlet_Sales` based on information such as item weight, item visibility, item type, MRP, outlet size, location, and outlet type.

The project includes data preprocessing, missing-value handling, categorical encoding, feature scaling, model training, evaluation, and prediction.

## Dataset

The dataset contains information about products and Big Mart outlets.

### Features

* Item_Identifier
* Item_Weight
* Item_Fat_Content
* Item_Visibility
* Item_Type
* Item_MRP
* Outlet_Identifier
* Outlet_Establishment_Year
* Outlet_Size
* Outlet_Location_Type
* Outlet_Type

### Target

* `Item_Outlet_Sales`

## Data Preprocessing

The following preprocessing steps were performed:

* Removed `Item_Identifier` from the features
* Filled missing `Item_Weight` values using median imputation
* Filled missing `Outlet_Size` values using most-frequent imputation
* Applied One-Hot Encoding to categorical features
* Applied StandardScaler to numerical features
* Split the dataset into training and testing sets using an 80/20 split

## Models

Several regression models were tested:

* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

### Best Model

The XGBoost Regressor achieved the best performance.

| Metric   |   Score |
| -------- | ------: |
| MAE      |  716.87 |
| RMSE     | 1029.30 |
| R² Score |  0.6102 |

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Pickle

## Project Structure

```text
Big-Mart-Sales-Prediction/
│
├── main.py
├── big_mart_model.pkl
├── encoder.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd Big-Mart-Sales-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run main.py
```

## Prediction

The application takes product and outlet information as input and predicts the expected sales for that product at the selected outlet.

## Future Improvements

* Hyperparameter tuning
* Feature engineering
* Improved model performance
* Better Streamlit UI
* Model deployment

## Author

Rida
