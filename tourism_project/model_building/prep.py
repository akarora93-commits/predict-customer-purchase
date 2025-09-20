# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/akarora93/predict-customer-purchase/predict-customer-purchase.csv"
customer_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'Exited'

# List of numerical features in the dataset
numeric_features = [
    'Age',                        # Customer's age
    'CityTier',                   # Tier of the city
    'DurationOfPitch',            # Duration of sales pitch
    'NumberOfPersonVisiting',     # Number of people visiting together
    'NumberOfFollowups',          # Number of follow-ups
    'PreferredPropertyStar',      # Preferred property star rating
    'NumberOfTrips',              # Number of previous trips
    'Passport',                   # Binary feature (0 or 1)
    'PitchSatisfactionScore',     # Satisfaction score of pitch
    'OwnCar',                     # Binary feature (0 or 1)
    'NumberOfChildrenVisiting',   # Number of children visiting
    'MonthlyIncome'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',    # Self Enquiry / Company Invited
    'Occupation',       # Salaried / Small Business / Free Lancer
    'Gender',           # Male / Female
    'ProductPitched',   # Basic / Deluxe / etc.
    'MaritalStatus',    # Single / Married / Divorced
    'Designation'       # Executive / Manager / etc.
]

# Define target variable
target = 'ProdTaken'   # 1 = Purchased, 0 = Not Purchased

# Define predictor matrix (X) using selected numeric and categorical features
X = customer_dataset[numeric_features + categorical_features]

# Define target vector (y)
y = customer_dataset[target]

# Split dataset into train and test
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility
)

# Save splits to CSV
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload files to Hugging Face Hub (update <HF-UserID>)
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="akarora93/predict-customer-purchase",
        repo_type="dataset",
    )
