#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import sagemaker
import boto3
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput

# Load your dataset
file_path = 's3://climatechanges98/climatedata/Cleaned_GlobalTemperatures.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Check data types and null values
print(data.info())



# In[32]:


# Convert 'dt' to datetime format and extract year and month
data['dt'] = pd.to_datetime(data['dt'])
data['year'] = data['dt'].dt.year
data['month'] = data['dt'].dt.month

# Check for missing values in 'LandAverageTemperature'
missing_values = data['LandAverageTemperature'].isnull().sum()

# Updated data head and check missing values
data.head(), missing_values


# In[33]:


from sklearn.model_selection import train_test_split
# Normalize the 'year' feature to have a mean of 0 and a standard deviation of 1
year_mean = data['year'].mean()
year_std = data['year'].std()
data['year_normalized'] = (data['year'] - year_mean) / year_std
# Selecting features and target
features = data[['year_normalized', 'month']]
target = data['LandAverageTemperature']

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Checking the shapes of the splits
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[38]:


year_mean


# In[39]:


year_std


# In[34]:


import sagemaker
import os
from sagemaker.session import Session

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Get execution role for SageMaker
role = sagemaker.get_execution_role()

# Prepare the directory and save training and validation data as CSV
os.makedirs("sagemaker", exist_ok=True)
train_path = 'sagemaker/train.csv'
val_path = 'sagemaker/validation.csv'

# Save to CSV
pd.concat([y_train, X_train], axis=1).to_csv(train_path, header=False, index=False)
pd.concat([y_val, X_val], axis=1).to_csv(val_path, header=False, index=False)

# Upload data to S3
train_data_s3 = sagemaker_session.upload_data(path=train_path, key_prefix='data')
val_data_s3 = sagemaker_session.upload_data(path=val_path, key_prefix='data')


# In[ ]:





# In[35]:


import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Get execution role
role = get_execution_role()

# Get the XGBoost image URI
xgboost_container = sagemaker.image_uris.retrieve('xgboost', sagemaker_session.boto_region_name, '1.0-1')

# Create a SageMaker Estimator
xgb = sagemaker.estimator.Estimator(xgboost_container,
                                    role,
                                    instance_count=1,
                                    instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/output'.format(sagemaker_session.default_bucket()),
                                    sagemaker_session=sagemaker_session)

# Set hyperparameters (use 'reg:linear' for older versions of XGBoost)
xgb.set_hyperparameters(objective='reg:linear',  # Updated here
                        num_round=100,
                        max_depth=10,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.7)


# In[36]:


data_channels = {
    'train': sagemaker.inputs.TrainingInput(train_data_s3, content_type='text/csv'),
    'validation': sagemaker.inputs.TrainingInput(val_data_s3, content_type='text/csv')
}


# In[37]:


xgb.fit(data_channels)


# In[40]:


custom_endpoint_name = "xgboostlinear-endpoint"  # Define your custom endpoint name
predictor = xgb.deploy(initial_instance_count=1,
                          instance_type='ml.m4.xlarge',
                          endpoint_name=custom_endpoint_name)


# In[30]:


import boto3
import sagemaker
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Create a SageMaker runtime client
client = boto3.client('sagemaker-runtime')

# Specify your endpoint name
endpoint_name = 'xgboost-endpoint1'

# Prepare the input data - let's say we want to predict the temperature for January 2020
input_data = '2030,9'

# Serialize the input data
serializer = CSVSerializer()

# Use the SageMaker runtime client to invoke the endpoint
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='text/csv',
    Body=serializer.serialize(input_data)
)

# Deserialize the response
deserializer = JSONDeserializer()
result = deserializer.deserialize(response['Body'], content_type='application/json')

print("Predicted LandAverageTemperature:", result)


# In[47]:


import boto3
import sagemaker
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Constants for year normalization (replace these with your actual computed values from the training dataset)
YEAR_MEAN = 1932.5  # Example mean year of the training dataset
YEAR_STD = 47.9    # Example standard deviation of the training dataset

# Create a SageMaker runtime client
client = boto3.client('sagemaker-runtime')

# Specify your endpoint name
endpoint_name = 'xgboostlinear-endpoint'

# Prepare the input data - let's say we want to predict the temperature for September 2034
year = 2070
month = 9

# Normalize the year
normalized_year = (year - YEAR_MEAN) / YEAR_STD

# Prepare the input data
input_data = f"{normalized_year},{month}"

# Serialize the input data
serializer = CSVSerializer()

# Use the SageMaker runtime client to invoke the endpoint
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='text/csv',
    Body=serializer.serialize(input_data)
)

# Deserialize the response
deserializer = JSONDeserializer()
result = deserializer.deserialize(response['Body'], content_type='application/json')

print("Predicted LandAverageTemperature:", result)


# In[ ]:


Predicted LandAverageTemperature: 12.916250228881836

