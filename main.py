import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle

# -------------------------Reading the data on Latitude and Longitude of all the MRT Stations in Singapore------------------------
data = pd.read_csv('df_coordinates.csv')

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon="🏨",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "book"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )

# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :red[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :red[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :red[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :red[Domain :] Real Estate")

# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :red[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price]")

    try:
        with st.form("form1"):

            # -----New Data inputs from the user for predicting the resale price-----
            street_name = st.selectbox("Street Name", data['address'].unique())
            block = st.selectbox("Block Number", data['block_no'].unique())
            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
            lease_commence_date = st.number_input('Lease Commence Date', min_value=1960, max_value=2023)
            storey_range = st.text_input("Storey Range (Format: '1 TO 2')")

            # -----Submit Button for PREDICT RESALE PRICE-----
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button:
                with open("model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open('scaler.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)

                # -----Calculating lease_remain_years using lease_commence_date-----
                lease_remain_years = 99 - (2023 - lease_commence_date)

                # -----Calculating median of storey_range to make our calculations quite comfortable-----
                split_list = storey_range.split(' TO ')
                float_list = [float(i) for i in split_list]
                storey_median = statistics.median(float_list)

                # -----Getting the address by joining the block number and the street name-----
                address = block + " " + street_name
                query_address = address
                query_string = f'https://www.onemap.gov.sg/api/common/elastic/search?searchVal={query_address}&returnGeom=Y&getAddrDetails=Y'
                resp = requests.get(query_string)

                # -----Using OpenMap API getting the latitude and longitude location of that address-----
                origin = []
                data_geo_location = json.loads(resp.content)
                if data_geo_location['found'] != 0:
                    latitude = data_geo_location['results'][0]['LATITUDE']
                    longitude = data_geo_location['results'][0]['LONGITUDE']
                    origin.append((latitude, longitude))

                # -----Appending the Latitudes and Longitudes of the MRT Stations-----
                # Latitudes and Longitudes are been appended in the form of a tuple  to that list
                mrt_lat = data['latitude']
                mrt_long = data['longitude']
                list_of_mrt_coordinates = []
                for lat, long in zip(mrt_lat, mrt_long):
                    list_of_mrt_coordinates.append((lat, long))

                # -----Getting distance to nearest MRT Stations (Mass Rapid Transit System)-----
                list_of_dist_mrt = []
                for destination in range(len(list_of_mrt_coordinates)):
                    list_of_dist_mrt.append(geodesic(origin, list_of_mrt_coordinates[destination]).meters)
                min_dist_mrt = min(list_of_dist_mrt)

                # -----Getting distance from CDB (Central Business District)-----
                cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

                # -----Sending the user enter values for prediction to our model-----
                new_sample = np.array(
                    [[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                new_sample = scaler_loaded.transform(new_sample)
                new_pred = loaded_model.predict(new_sample)[0]
                st.write('## :green[Predicted resale price: $] ', np.exp(new_pred))

    except Exception as e:
        st.write("Error occurred:", e)
        st.write("Enter the above values to get the predicted resale price of the flat")
