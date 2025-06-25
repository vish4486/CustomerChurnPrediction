import os
import pytest
import pandas as pd
from src.data_preprocessing import load_and_inspect_data
from src.model_train import evaluate_model_advanced
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

@pytest.fixture
def sample_dataframe():
    filepath = "data/raw/BankChurners.csv"
    assert os.path.exists(filepath), "Dataset file not found."
    df = load_and_inspect_data(filepath)
    return df

def test_load_and_inspect_data(sample_dataframe):
    assert isinstance(sample_dataframe, pd.DataFrame)
    assert "Attrition_Flag" in sample_dataframe.columns
    assert sample_dataframe.shape[0] > 0

def test_evaluate_model_advanced(sample_dataframe):
    sample_dataframe = sample_dataframe.drop(columns=['CLIENTNUM'])
    sample_dataframe['Attrition_Flag'] = sample_dataframe['Attrition_Flag'].map({
        'Existing Customer': 0,
        'Attrited Customer': 1
    })
    
    X = sample_dataframe.drop("Attrition_Flag", axis=1).select_dtypes(include='number')
    y = sample_dataframe["Attrition_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    result = evaluate_model_advanced("TestLogReg", model, X_train, y_train, X_test, y_test)

    required_keys = [
        'Model', 'Balanced Accuracy', 'F1 Score', 'Precision', 'Recall (TPR)',
        'Specificity (TNR)', 'FPR', 'FNR', 'ROC AUC', 'Training Time (s)'
    ]
    for key in required_keys:
        assert key in result

