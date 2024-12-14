from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('house_price_model.pkl')

# Read the cleaned data
df = pd.read_csv('Data.csv')

# Get the unique locations for dropdown
locations = sorted(df['Location'].unique())

# Home route for rendering the form
@app.route('/')
def index():
    return render_template('index.html', locations=locations)

# Route for predicting house price
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        location = request.form['location']

        if location == 'Select Location':
            return jsonify({'error': 'Please select a valid location'})

        # Get latitude and longitude for the selected location
        latitude = df[df['Location'] == location]['Latitude'].values[0]
        longitude = df[df['Location'] == location]['Longitude'].values[0]

        # Prepare the input array
        X_custom = np.array([[area, bedrooms, latitude, longitude]])

        # Predict the price
        y_custom_pred = model.predict(X_custom)
        predicted_price = round(y_custom_pred[0], 2)

        # Return the prediction
        return jsonify({'predicted_price': f'{predicted_price} Rupees'})

    except ValueError:
        return jsonify({'error': 'Please enter valid input values'})

        

# Main driver
if __name__ == "__main__":
    app.run(debug=True)
