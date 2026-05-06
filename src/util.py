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
        upper_limit_duration = df_copy[col].quantile(0.99)

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

def feature_engineering_pipeline(df):

    # 2. Temporal Features
    if 'Start_Time' in df.columns:
        df['Start_Time'] = pd.to_datetime(df['Start_Time'])
        df['Hour'] = df['Start_Time'].dt.hour
        df['Month'] = df['Start_Time'].dt.month
        df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
        
        # Binary flags
        df['Is_Rush_Hour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
        df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Cyclical encoding
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    if 'Sunrise_Sunset' in df.columns:
        df['Is_Night'] = (df['Sunrise_Sunset'] == 'Night').astype(int)

    # 3. Weather Features
    bad_weather_groups = ['Rain', 'Fog', 'Snow_Ice', 'Severe']
    if 'Weather_Group' in df.columns:
        df['Is_Bad_Weather'] = df['Weather_Group'].isin(bad_weather_groups).astype(int)
    
    if 'Visibility(mi)' in df.columns:
        df['Low_Visibility_Flag'] = (df['Visibility(mi)'] < 2).astype(int)
    
    if 'Temperature(F)' in df.columns:
        df['Freezing_Flag'] = (df['Temperature(F)'] < 32).astype(int)
        
    weather_risk_cols = ['Is_Bad_Weather', 'Low_Visibility_Flag', 'Freezing_Flag']
    df['Weather_Risk_Score'] = df[[c for c in weather_risk_cols if c in df.columns]].sum(axis=1)

    # 4. Infrastructure Features
    infra_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
                  'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
                  'Traffic_Calming', 'Traffic_Signal']
    existing_infra = [c for c in infra_cols if c in df.columns]
    if existing_infra:
        df['Road_Complexity'] = df[existing_infra].astype(int).sum(axis=1)
    
    if 'Crossing' in df.columns and 'Junction' in df.columns:
        df['Is_Intersection'] = (df['Crossing'].astype(int) | df['Junction'].astype(int)).astype(int)
    
    if 'Traffic_Signal' in df.columns and 'Stop' in df.columns:
        df['Is_Controlled'] = (df['Traffic_Signal'].astype(int) | df['Stop'].astype(int)).astype(int)

    # 5. Spatial Features (Frequency Encoding)
    if 'State' in df.columns:
        state_freq = df['State'].value_counts(normalize=True)
        df['State_Freq'] = df['State'].map(state_freq)
    
    if 'City' in df.columns:
        city_freq = df['City'].value_counts(normalize=True)
        df['City_Freq'] = df['City'].map(city_freq)

    # 6. Description Keywords (Insight from EDA)
    if 'Description' in df.columns:
        desc = df['Description'].str.lower()
        df['is_shoulder'] = desc.str.contains('shoulder', na=False).astype(int)
        df['is_blocked'] = desc.str.contains('blocked', na=False).astype(int)
        df['is_overturned'] = desc.str.contains('overturned', na=False).astype(int)

    # 7. Interaction Terms
    if 'Severity' in df.columns:
        df['Rush_x_Severity'] = df['Is_Rush_Hour'] * df['Severity']
        df['Night_x_Severity'] = df.get('Is_Night', 0) * df['Severity']

    return df
