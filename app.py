import streamlit as st
import pandas as pd
from keras.models import model_from_json

# File paths to the model's JSON and weights
model_json_path = r'c:\Users\Sharvari Gohane\Documents\fake-social-media-profile-detection\model.json'
model_weights_path = r'c:\Users\Sharvari Gohane\Documents\fake-social-media-profile-detection\model_json.weights.h5'

# Load JSON and create model
with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights(model_weights_path)

# Streamlit input fields
stst_count = st.number_input("Enter Status Count:", min_value=0, value=0, step=1)
follower_count = st.number_input("Enter Follower Count:", min_value=0, value=0, step=1)
following_count = st.number_input("Enter Following Count:", min_value=0, value=0, step=1)
fav_count = st.number_input("Enter Favourites Count:", min_value=0, value=0, step=1)
listed_count = st.number_input("Enter Listed Count:", min_value=0, value=0, step=1)
geo_enable = st.selectbox("Enter Geo Enabled Count [0,1]:", options=[0, 1])
profile = st.selectbox("Enter Profile Count [0,1]:", options=[0, 1])

# Prepare the input data for prediction
prediction_df = pd.DataFrame([{
    "statuses_count": stst_count,
    "followers_count": follower_count,
    "friends_count": following_count,
    "favourites_count": fav_count,
    "lang_num": 1,
    "listed_count": listed_count,
    "geo_enabled": geo_enable,
    "profile_use_background_image": profile
}])

# Make prediction when the button is clicked
if st.button('Predict'):
    st.write("Input data for prediction:", prediction_df)
    prediction = model.predict(prediction_df)
    prediction = prediction[0][0]  # Adjust indexing based on your model output shape
    
    if prediction > 0.5:
        st.write("Prediction: Fake Profile")
    else:
        st.write("Prediction: Real Profile")
