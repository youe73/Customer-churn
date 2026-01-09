import pytest
import pandas as pd
import numpy as np
from src import train_model

def test_cleaning_transform():
    df_train = pd.read_csv("./src/customer_churn_dataset-training-master.csv")
    df_test =  pd.read_csv("./src/customer_churn_dataset-testing-master.csv")
    
    mergeddf = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    mergeddf_new = mergeddf.dropna()
    cleaned =train_model.cleaning_transform(mergeddf_new)
    
    assert mergeddf.shape != cleaned.shape
    assert cleaned["Age"].dtype == int
    assert cleaned["Tenure"].dtype == int
    assert cleaned["Usage Frequency"].dtype == int
    assert cleaned["Support Calls"].dtype == int
    assert cleaned["Payment Delay"].dtype == int
    assert cleaned["Last Interaction"].dtype == int
    assert cleaned["Churn"].dtype == int
    assert cleaned["Subscription Type"].dtype == int
    assert cleaned["Contract Length"].dtype == int
    assert cleaned["Total Spend"].dtype == mergeddf["Total Spend"].dtype
    assert cleaned["Gen_Female"].dtype == int
    assert cleaned["Gen_Male"].dtype == int

    
     