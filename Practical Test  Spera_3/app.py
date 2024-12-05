from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('Models\\logistic_regression_model.pkl')

# Define the prediction function
def predict_churn(data):
    input_data = pd.DataFrame(data, index=[0])
    model_columns = model.feature_names_in_
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_data)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'gender': request.form['gender'],
        'SeniorCitizen': request.form['senior_citizen'],
        'Partner': request.form['partner'],
        'Dependents': request.form['dependents'],
        'tenure': float(request.form['tenure']),
        'PhoneService': request.form['phone_service'],
        'MultipleLines': request.form['multiple_lines'],
        'InternetService_DSL': 1 if request.form['internet_service'] == 'DSL' else 0,
        'InternetService_Fiber optic': 1 if request.form['internet_service'] == 'Fiber optic' else 0,
        'InternetService_No': 1 if request.form['internet_service'] == 'No' else 0,
        'OnlineSecurity': request.form['online_security'],
        'OnlineBackup': request.form['online_backup'],
        'DeviceProtection': request.form['device_protection'],
        'TechSupport': request.form['tech_support'],
        'StreamingTV': request.form['streaming_tv'],
        'StreamingMovies': request.form['streaming_movies'],
        'Contract_Month-to-month': 1 if request.form['contract'] == 'Month-to-month' else 0,
        'Contract_One year': 1 if request.form['contract'] == 'One year' else 0,
        'Contract_Two year': 1 if request.form['contract'] == 'Two year' else 0,
        'PaperlessBilling': request.form['paperless_billing'],
        'PaymentMethod_Bank transfer (automatic)': 1 if request.form['payment_method'] == 'Bank transfer' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if request.form['payment_method'] == 'Credit card' else 0,
        'PaymentMethod_Electronic check': 1 if request.form['payment_method'] == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if request.form['payment_method'] == 'Mailed check' else 0,
        'MonthlyCharges': float(request.form['monthly_charges']),
        'TotalCharges': float(request.form['total_charges'])
    }

    prediction = predict_churn(data)
    return render_template('index.html', prediction_text=f'Churn Prediction: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
