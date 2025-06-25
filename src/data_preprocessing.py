import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.model_utils import save_plot
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_and_inspect_data(filepath):
    df = pd.read_csv(filepath)
    print("Shape of dataset:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nChurn Distribution:\n", df['Attrition_Flag'].value_counts(normalize=True) * 100)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescriptive Stats:\n", df.describe(include='all'))
    print("\nData Types:\n", df.dtypes)
    return df

def visualize_churn_distribution(df, save=False):
    fig, ax = plt.subplots()
    sns.countplot(x='Attrition_Flag', data=df, ax=ax)
    ax.set_title('Class Distribution: Churn vs Non-Churn')
    ax.set_ylabel('Count')
    ax.set_xlabel('Customer Type')
    plt.xticks(rotation=15)

    if save:
        save_plot(fig, "churn_distribution.png", subfolder="plots")
    else:
        plt.show()

def visualize_correlations(df, save=False):
    drop_cols = [
        'CLIENTNUM', 'Attrition_Flag',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    df_corr = df.drop(columns=drop_cols)
    df_corr = df_corr.select_dtypes(include=['int64', 'float64'])

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)
    ax.set_title("Feature Correlation Heatmap")

    if save:
        save_plot(fig, "feature_correlation_heatmap.png", subfolder="plots")
    else:
        plt.show()

def preprocess_data(df):
    os.makedirs("data/processed", exist_ok=True)

    df = df.drop(columns=[
        'CLIENTNUM',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ])

    df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).drop('Attrition_Flag', axis=1).columns.tolist()

    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Add interaction feature
    X_train['Transaction_Intensity'] = X_train['Total_Trans_Ct'] * X_train['Total_Amt_Chng_Q4_Q1']
    X_test['Transaction_Intensity'] = X_test['Total_Trans_Ct'] * X_test['Total_Amt_Chng_Q4_Q1']

    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.append('Transaction_Intensity')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    encoded_cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    processed_columns = list(numerical_cols) + list(encoded_cat_columns)

    #  Save processed components to disk
    joblib.dump(X_train_transformed, "data/processed/X_train.pkl")
    joblib.dump(X_test_transformed, "data/processed/X_test.pkl")
    joblib.dump(y_train, "data/processed/y_train.pkl")
    joblib.dump(y_test, "data/processed/y_test.pkl")
    joblib.dump(preprocessor, "data/processed/preprocessor.pkl")
    joblib.dump(processed_columns, "data/processed/columns.pkl")

    print("Preprocessing Complete (with interaction feature)")
    print(f"X_train shape: {X_train_transformed.shape}")
    print(f"X_test shape: {X_test_transformed.shape}")

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, processed_columns

def perform_additional_eda(df):
    
    os.makedirs("plots", exist_ok=True)
    # Numerical histograms
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Histogram of {col}')
        save_plot(fig, f"hist_{col}.png", subfolder="plots")

    # Boxplots for outliers
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Boxplot of {col}')
        save_plot(fig, f"box_{col}.png", subfolder="plots")

    # Categorical bar plots
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Bar Chart of {col}')
        save_plot(fig, f"bar_{col}.png", subfolder="plots")

    # Check duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")

    # Check inconsistent values (sample)
    for col in cat_cols:
        print(f"\nUnique values in '{col}': {df[col].unique()}")
