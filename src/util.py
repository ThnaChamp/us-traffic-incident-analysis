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

def apply_weather_grouping(df):
    df_copy = df.copy()
    
    # Define weather keyword groups based on frequent words
    weather_keywords = {
        'Severe': 'thunder|storm|tornado|gale|hail|squalls|funnel|whirl|volcanic',
        'Snow_Ice': 'snow|sleet|ice|freez|wintry|mix|pellet',
        'Fog': 'fog|haze|smoke|dust|mist|patches|sand|ash|shallow',
        'Rain': 'rain|drizzle|shower|precip',
        'Windy': 'wind|breezy|blustery|gusty',
        'Cloudy': 'cloud|overcast',
        'Clear': 'clear|fair'
    }
    
    if 'Weather_Condition' in df_copy.columns:
        weather_series = df_copy['Weather_Condition'].str.lower().fillna('')
        
        # Priority order: Severe > Snow_Ice > Fog > Rain > Windy > Cloudy > Clear
        conditions = [
            weather_series.str.contains(weather_keywords['Severe'], na=False),
            weather_series.str.contains(weather_keywords['Snow_Ice'], na=False),
            weather_series.str.contains(weather_keywords['Fog'], na=False),
            weather_series.str.contains(weather_keywords['Rain'], na=False),
            weather_series.str.contains(weather_keywords['Windy'], na=False),
            weather_series.str.contains(weather_keywords['Cloudy'], na=False),
            weather_series.str.contains(weather_keywords['Clear'], na=False)
        ]
        
        choices = ['Severe', 'Snow_Ice', 'Fog', 'Rain', 'Windy', 'Cloudy', 'Clear']
        df_copy['Weather_Group'] = np.select(conditions, choices, default='Other')
    else:
        if 'Weather_Group' not in df_copy.columns:
            df_copy['Weather_Group'] = 'Other'

    # Fallback logic for 'Other' or NaN using numeric columns
    mask_other = df_copy["Weather_Group"].isin(["Other", np.nan])
    
    if mask_other.any():
        # Ensure required columns exist for fallback
        for col in ['Wind_Speed(mph)', 'Precipitation(in)', 'Temperature(F)', 'Visibility(mi)', 'Humidity(%)']:
            if col not in df_copy.columns:
                df_copy[col] = np.nan

        conditions_from_numbers = [
            (((df_copy['Wind_Speed(mph)'].notnull()) & (df_copy['Wind_Speed(mph)'] > 40)) | 
             ((df_copy['Precipitation(in)'].notnull()) & (df_copy['Precipitation(in)'] > 1.0))),

            (df_copy['Precipitation(in)'].notnull()) & (df_copy['Temperature(F)'].notnull()) & 
            (df_copy['Precipitation(in)'] > 0) & (df_copy['Temperature(F)'] < 32),
            
            (df_copy['Precipitation(in)'].notnull()) & (df_copy['Temperature(F)'].notnull()) & 
            (df_copy['Precipitation(in)'] > 0) & (df_copy['Temperature(F)'] >= 32),
            
            (df_copy['Visibility(mi)'].notnull()) & 
            (df_copy['Visibility(mi)'] < 2.0),
            
            (df_copy['Wind_Speed(mph)'].notnull()) & 
            (df_copy['Wind_Speed(mph)'] > 20),
            
            (df_copy['Humidity(%)'].notnull()) & 
            (df_copy['Humidity(%)'] > 80),
            
            (df_copy['Temperature(F)'].notnull())
        ]

        choices_for_numbers = ["Severe", "Snow_Ice", "Rain", "Fog", "Windy", "Cloudy", "Clear"]
        
        df_copy.loc[mask_other, "Weather_Group"] = np.select(
            [cond[mask_other] for cond in conditions_from_numbers],
            choices_for_numbers,
            default="Other",
        )
        
    return df_copy
