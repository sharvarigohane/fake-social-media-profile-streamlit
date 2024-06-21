import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
from keras.models import Sequential, model_from_json
import pandas as pd
import numpy as np

# Full path to model files
model_json_path = r'c:\Users\Sharvari Gohane\Documents\fake-social-media-profile-detection\model.json'
model_weights_path = r'c:\Users\Sharvari Gohane\Documents\fake-social-media-profile-detection\model_json.weights.h5'

# Load JSON and create model
json_file = open(model_json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Input data
prediction_df = pd.DataFrame([{
    "statuses_count": 1,
    "followers_count": 554,
    "friends_count": 534,
    "favourites_count": 0,
    "lang_num": 1,
    "listed_count": 0,
    "geo_enabled": 1,
    "profile_use_background_image": 1
}])

# Convert DataFrame to NumPy array
input_data = prediction_df.to_numpy()

print("Input Data:")
print(prediction_df)

# Make prediction
prediction = loaded_model.predict(input_data)
prediction = prediction[0][0]  # Assuming the model output is (1, 1) shaped

print('Prediction:', prediction)
if prediction > 0.5:
    print("fake profile")
else:
    print("real profile")