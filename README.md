# Retail-Sales-Data-Analysis
This project performs a complete data preprocessing and regression analysis workflow on a retail sales dataset. It includes data cleaning, imputation, categorical encoding, feature scaling, and building an Ordinary Least Squares (OLS) regression model to understand how pricing affects total sales. 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
import matplotlib.pyplot as plt

def main():
    # Load dataset (replace with your actual file path)
    try:
        df = pd.read_csv('/content/retail_sales_dataset.csv')
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        return

    print("\nInitial data:")
    print(df.head(8))
    print(f"\nTotal rows and columns before cleaning: {df.shape}")

    # Data Cleaning - Option 1: Drop missing values
    df_cleaned = df.dropna()
    print("\nData after dropping missing values:")
    print(df_cleaned.head(8))
    print(f"Rows/columns after dropping: {df_cleaned.shape}")

    # Data Cleaning - Option 2: Imputation
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = ['Age', 'Quantity', 'Price per Unit', 'Total Amount']
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    print("\nData after imputation:")
    print(df.head(8))
    print(f"Rows/columns after imputation: {df.shape}")

    # Categorical Encoding
    # Label Encoding
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    print("\nAfter Label Encoding (Gender):")
    print(df[['Gender']].head())

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Product Category'], drop_first=True)
    print("\nAfter One-Hot Encoding (Product Category):")
    print(df.head())

    # Feature Scaling
    # Min-Max Scaling
    minmax_cols = ['Quantity', 'Price per Unit', 'Total Amount']
    scaler = MinMaxScaler()
    df[minmax_cols] = scaler.fit_transform(df[minmax_cols])
    print("\nAfter Min-Max Scaling:")
    print(df[minmax_cols].head())

    # Standardization
    std_cols = ['Age', 'Quantity', 'Price per Unit', 'Total Amount']
    std_scaler = StandardScaler()
    df[std_cols] = std_scaler.fit_transform(df[std_cols])
    print("\nAfter Standardization:")
    print(df[std_cols].head())

    # OLS Regression
    X = df[['Price per Unit']]
    y = df['Total Amount']
    X = sm.add_constant(X)  # Add intercept term

    model = sm.OLS(y, X).fit()
    print("\n" + "="*50)
    print("REGRESSION RESULTS")
    print("="*50)
    print(model.summary())

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Price per Unit'], df['Total Amount'],
                alpha=0.5, label='Actual Data')
    plt.plot(df['Price per Unit'], model.predict(X),
             color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Price per Unit')
    plt.ylabel('Total Amount')
    plt.title('Price vs. Total Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
