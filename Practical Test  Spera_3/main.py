from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def preprocess_input(data):
    # Define the categorical features that need encoding
    categorical_features = [
        'gender', 'Partner', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    
    # Perform one-hot encoding
    data = pd.get_dummies(data, columns=categorical_features)
    
    # Ensure all expected columns are present
    expected_columns = [
        'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'gender_Female', 'gender_Male',
        'Partner_No', 'Partner_Yes',
        'MultipleLines_No', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_Yes', 
        'OnlineBackup_No', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_Yes',
        'TechSupport_No', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_Yes',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    for column in expected_columns:
        if column not in data.columns:
            data[column] = 0  # Add missing columns with default value 0

    data = data[expected_columns]  # Ensure correct column order

    # Scale numeric features
    scaler = MinMaxScaler()
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[scale_cols] = scaler.fit_transform(data[scale_cols])

    return data
