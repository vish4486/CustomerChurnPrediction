import pandas as pd
import pytest
from src.data_preprocessing import preprocess_data

@pytest.fixture
def sample_dataframe():
    # Mocked minimal dataset
    data = {
        'Attrition_Flag': ['Existing Customer', 'Attrited Customer'] * 3,
        'Customer_Age': [45, 50, 35, 60, 40, 55],
        'Total_Trans_Ct': [42, 55, 30, 65, 38, 72],
        'Total_Amt_Chng_Q4_Q1': [1.2, 1.1, 0.8, 1.5, 1.0, 0.9],
        'Gender': ['M', 'F', 'F', 'M', 'M', 'F'],
        'Income_Category': ['$60K - $80K', 'Less than $40K', 'Less than $40K', '$80K - $120K', '$60K - $80K', '$40K - $60K']
    }
    return pd.DataFrame(data)

def test_columns_present(sample_dataframe):
    assert 'Attrition_Flag' in sample_dataframe.columns
    assert 'Total_Trans_Ct' in sample_dataframe.columns

def test_preprocessing_output_shape(sample_dataframe):
    X_train, X_test, y_train, y_test, preprocessor, columns = preprocess_data(sample_dataframe)
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]

