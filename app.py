from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn import preprocessing

app = Flask(__name__, template_folder='pages')

# Load the trained Random Forest model
rf_model = joblib.load('./model/xgboost_model_new.pkl')

# Define a route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions

# Define a route for making predictions
@app.route('/predictmod', methods=['POST'])
def predict_mod():
    # Ensure the request content type is JSON
    if request.headers['Content-Type'] != 'application/json':
        return jsonify({'error': 'Unsupported Media Type'}), 415

    # Get the request JSON data
    request_data = request.json

     # Convert JSON data to DataFrame, excluding column names
    data = {row['name']: float(row['value']) for row in request_data}
    df = pd.DataFrame([data])


    # Convert JSON data to DataFrame
    #df = pd.DataFrame.from_dict(request_data, orient='columns')

    

    # Preprocess the input data (normalize features)
    normalizer = preprocessing.StandardScaler()
    df_scaled = normalizer.fit_transform(df)
   

    # Make predictions using the loaded model
    predictions = rf_model.predict(df_scaled)


    print(predictions)
    # Dictionary mapping bean types to their corresponding indices
    bean_mapping = {
    0: 'BARBUNYA',
    1: 'BOMBAY',
    2: 'CALI',
    3: 'DERMASON',
    4: 'HOROZ',
    5: 'SEKER',
    6: 'SIRA'
    }

    # Map predictions to bean types
    predicted_beans = [bean_mapping[prediction] for prediction in predictions]

    # Return predictions as JSON response
    #return jsonify({'predictions': predictions.tolist()}), 200
    return jsonify({'predictions': predicted_beans}), 200

if __name__ == '__main__':
    app.run(debug=True)
