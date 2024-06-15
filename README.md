# Singapore Resale Flat Prices Prediction

This project aims to build a machine learning model and implement it as a user-friendly online application to predict the resale values of apartments in Singapore. The prediction model is based on past transactions involving resale flats and is designed to help both future buyers and sellers evaluate the worth of a flat after it has been previously resold.

## Table of Contents
- [Technologies](#technologies)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Model Description](#model-description)

## Technologies
- Python
- Pandas
- Numpy
- Scikit-Learn
- Streamlit
- Machine Learning
- Data Preprocessing
- Visualization
- EDA
- Model Building
- Data Wrangling
- Model Deployment

## Overview
This project leverages historical data on resale flat transactions to build a predictive model for future resale prices. Factors such as location, apartment type, floor area, and lease duration influence resale prices. The model provides users with an expected resale price based on these criteria, assisting in making informed decisions.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/resale-flat-prices-prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd resale-flat-prices-prediction
    ```

3. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```


## Usage
To start the Streamlit application, run the following command:
```bash
streamlit run app.py
```


## Input Fields
- Street Name: Select from the dropdown list of street names.
- Block Number: Select from the dropdown list of block numbers.
- Floor Area (Per Square Meter): Enter the floor area in square meters.
- Lease Commence Date: Enter the lease commence date.
- Storey Range: Enter the storey range in the format '1 TO 2'.
- Click on the Predict Resale Price button to get the predicted resale price.

## Files 
- main.py: The main application file for Streamlit.
- ML__Model.ipynb : Python file foe ML modelling after pre-processing the data.
- resale_preprocessing.ipynb : Python file for pre-processing the data
- df_coordinates.csv: Contains latitude and longitude data of MRT Stations in Singapore.
- model.pkl: The trained machine learning model.
- scaler.pkl: The scaler used for preprocessing the input data.


## Model Description 
The predictive model is built using the XGBoost regressor. The features used for prediction include:

Distance to the nearest MRT Station: Calculated using geodesic distance from the input address to the MRT stations.
Distance to the Central Business District (CBD): Calculated using geodesic distance from the input address to the CBD coordinates (1.2830, 103.8513).
Floor area in square meters: The total area of the flat.
Remaining lease years: Calculated based on the lease commence date provided by the user.
Median storey range: Calculated from the storey range input by the user.
The model was trained on historical transaction data and optimized using GridSearchCV for hyperparameter tuning. The model and scaler used for preprocessing are saved as model.pkl and scaler.pkl, respectively.
