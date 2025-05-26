import os
import tempfile
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    #Read Data
    df = pd.read_csv(
        f"{base_dir}/input/mob_price_classification.csv"
    )

    features = list(df.columns)
    label = features.pop(-1)

    # Store features and labels in two dataframes x and y respectively
    x = df[features]
    y = df[label]

    #Train test split
    X_train, X_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=0)

    # Create Train and Test dataframes to be stored for further use
    trainX = pd.DataFrame(X_train)
    trainX[label] = y_train
    
    testX = pd.DataFrame(X_test)
    testX[label] = y_test
    
    train_df = X_train.copy()
    train_df[label] = y_train
    
    # Sample 20% of the training data
    baseline_df = train_df.sample(frac=0.2, random_state=42)
    
    # Save the Dataframes as csv files
    trainX.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    baseline_df.to_csv(f"{base_dir}/baseline/baseline.csv", header=True, index=False) #maybe header true
    testX.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)