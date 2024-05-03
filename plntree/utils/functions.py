import pandas as pd


def remove_outliers_iqr(df, column_name, threshold=1.5):
    """
    Remove outliers from a DataFrame using the IQR method.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column containing the data.
    - threshold (float): IQR multiplier to determine the range for outliers.

    Returns:
    - pd.DataFrame: DataFrame without outliers.
    """
    # Calculate Q1 and Q3
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)

    # Calculate IQR
    iqr = q3 - q1

    # Define the lower and upper bounds to identify outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Filter out outliers
    df_no_outliers = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return df_no_outliers
