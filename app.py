import json
import boto3
import numpy as np
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyswarm import pso
import joblib
import os
import tempfile

# Initialize clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
s3 = boto3.client('s3')

# Specify the bucket and object key for the .joblib file
bucket_name = 'sagemaker-us-east-2-533267341824'
joblib_file_key = 'model2.joblib'
output_bucket_name = 'sagemaker-us-east-2-533267341824'  # Specify the bucket where you want to save the result file

# Endpoint name of your DNN model on SageMaker
sagemaker_endpoint_name = 'tensorflow-inference-2024-10-01-01-14-53-252'

def preprocess_data(data):
    df = pd.DataFrame(data['data'])
    df = df.drop(columns=['year', 'month', 'day', 'weekofyear', 'hour', 'minute', 'gen_Sol'])
    np.random.seed(42)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    synthetic_data = []
    
    for i in range(1000):
        synthetic_row = df.iloc[i % len(df)].copy()
        fluctuation = np.random.normal(0, 0.01, size=len(numeric_columns))
        synthetic_row[numeric_columns] += fluctuation
        synthetic_row['use_HO'] += np.random.normal(0, 0.01)
        synthetic_row['time'] = pd.to_datetime(synthetic_row['time']) + pd.Timedelta(minutes=i)
        synthetic_data.append(synthetic_row)
    
    synthetic_df = pd.DataFrame(synthetic_data)
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(synthetic_df[numeric_columns]), columns=numeric_columns)
    normalized_df['time'] = synthetic_df['time'].values
    return normalized_df

def create_lagged_features_with_padding(df):
    features = df.drop(columns=['use_HO', 'time'])
    X_lagged = []
    for i in range(1, len(features)):
        X_lagged.append(features.iloc[:i].values.flatten())
    X_lagged_padded = pad_sequences(X_lagged, padding='pre', dtype='float32')
    return np.array(X_lagged_padded)

def objective(appliance_usage, importance_weights):
    return np.sum(importance_weights * appliance_usage)

def optimize_appliance_usage_pso(appliance_usage, importance_weights, quantile=0.01):
    lb = np.minimum(np.quantile(appliance_usage, quantile), appliance_usage * 0.99)
    ub = appliance_usage
    lb = np.minimum(lb, ub - 1e-6)
    optimized_usage, fopt = pso(objective, lb, ub, args=(importance_weights,), swarmsize=100, maxiter=200)
    return optimized_usage

def lambda_handler(event, context):
    data = event  # Assuming data is received as JSON in the body of an API Gateway POST request
    normalized_df = preprocess_data(data)
    X_lagged = create_lagged_features_with_padding(normalized_df)
    X_last = X_lagged[-1]  # Last available set of features
    future_steps = 100  # Number of future time steps to predict

    # Predict using SageMaker DNN model
    payload = json.dumps({"instances": X_last.reshape(1, -len(X_last)).tolist()})
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=sagemaker_endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    result = json.loads(response['Body'].read().decode())
    predictions = np.array(result['predictions']).flatten()

    # Load RandomForest model from S3 and optimize predictions
    rf_model = joblib.load(joblib_file_key)
    importance_weights = np.where(rf_model.feature_importances_ == 0, 1e-6, 1 / rf_model.feature_importances_)

    optimized_usage = optimize_appliance_usage_pso(predictions, importance_weights)
    new_optimized_use_HO = rf_model.predict(optimized_usage.reshape(1, -1))[0]

    # Save result to a file and upload it to S3
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(json.dumps({'optimized_use_HO': new_optimized_use_HO}).encode('utf-8'))
        tmp_filename = tmp_file.name

    # Upload the file to S3
    file_key = 'results/optimized_result.json'
    s3.upload_file(tmp_filename, output_bucket_name, file_key)
    os.unlink(tmp_filename)  # Clean up the temporary file

    # Generate a presigned URL to access the file
    url = s3.generate_presigned_url('get_object', Params={'Bucket': output_bucket_name, 'Key': file_key}, ExpiresIn=3600)

    return {
        'statusCode': 200,
        'body': json.dumps({'url': url})
    }

