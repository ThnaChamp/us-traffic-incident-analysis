import numpy as np
import pandas as pd

# Transformation
def auto_transform_skewness(df):
    df_copy = df.copy()

    feature_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    
    for col in feature_cols:
        if col != "Severity":
            skew = df_copy[col].skew()
            
            # Right skewed
            if skew > 1.0:
                if df_copy[col].min() >= 0:
                    df_copy[col] = np.log1p(df_copy[col])
            elif skew > 0.5:
                if df_copy[col].min() >= 0:
                    df_copy[col] = np.sqrt(df_copy[col])
                
            # Left skewed
            elif skew < -0.5:
                df_copy[col] = np.square(df_copy[col])
    return df_copy

# Check Skewness
def skewed(df):
    feature_cols = df.columns
    for col in feature_cols:
        if df[col].dtype in ["int64", "float64"]:
            skew = df[col].skew()
            direction = "right (positive)" if skew > 0 else "left (negative)"
            if abs(skew) > 1:
                print(f"[SKEWED]     {col:<30} skew = {skew:+.3f}  →  highly skewed {direction}")
            elif abs(skew) > 0.5:
                print(f"[MODERATE]   {col:<30} skew = {skew:+.3f}  →  moderately skewed {direction}")
            else:
                print(f"[NORMAL]     {col:<30} skew = {skew:+.3f}  →  approximately symmetric")

# Manage Outlier
def manage_outlier(df, outlier_columns):
    df_copy = df.copy()
    
    for col in outlier_columns:
        upper_limit_duration = df_copy[col].quantile(0.999)

        df_copy[col] = np.where(
            df_copy[col] > upper_limit_duration,
            upper_limit_duration,
            df_copy[col]
        )
    return df_copy
