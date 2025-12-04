import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import openai
import json
from dotenv import load_dotenv
import os
from sklearn.utils import resample
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from uuid import uuid4
import re
from datetime import datetime
import dateutil.parser
from dateutil import tz

# ------------------ CONFIG ------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DATASETS = {
    "Dataset1": "LC_Card_data.csv",
    "Dataset2": "Visa_Japan_data_5000users.csv",
    "Dataset3": "sample.csv"
}

# ------------------ FASTAPI ------------------
app = FastAPI(title="RFM Segmentation API")

# ------------------ REQUEST MODELS ------------------
class AdditionalAttribute(BaseModel):
    id: str
    name: str
    category: str

class RFMWeights(BaseModel):
    recency_weight: Optional[float] = None
    frequency_weight: Optional[float] = None
    monetary_weight: Optional[float] = None

class SegmentRequest(BaseModel):
    selected_file_name: str
    start_date: str
    end_date: str
    uid_col: str
    recency_col: str
    frequency_col: Optional[str] = ""
    monetary_col: str
    date_col: Optional[str] = None
    additional_attributes: Optional[List[AdditionalAttribute]] = None
    rfm_weights: Optional[RFMWeights] = None
    k_value: Optional[int] = None
    
class DatasetRequest(BaseModel):
    dataset_name: str

# ------------------ ENHANCED DATE PREPROCESSING ------------------
def robust_date_parser(date_string):
    """
    Robust date parser that handles multiple formats including:
    - 2024-06-20 03:28:45
    - Sat Feb 02 12:50:00 IST 2019
    - 2024-06-20
    - 02/02/2019
    - And many others...
    """
    if pd.isna(date_string) or date_string is None or str(date_string).strip() == '':
        return None
    
    date_str = str(date_string).strip()
    
    # Try dateutil.parser first (handles most cases)
    try:
        return dateutil.parser.parse(date_str, fuzzy=False)
    except:
        pass
    
    # Try specific common formats
    common_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%m-%d-%Y',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%Y/%m/%d',
        '%d.%m.%Y',
        '%m.%d.%Y',
        '%Y.%m.%d',
        '%b %d %Y %H:%M:%S',
        '%a %b %d %H:%M:%S %Z %Y',  # Sat Feb 02 12:50:00 IST 2019
        '%a %b %d %H:%M:%S %Y',     # Sat Feb 02 12:50:00 2019
        '%d-%b-%Y',
        '%d/%b/%Y',
        '%Y%m%d',
        '%H:%M:%S %d-%m-%Y',
        '%H:%M:%S %d/%m/%Y',
    ]
    
    for fmt in common_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try parsing Unix timestamp
    try:
        if date_str.isdigit() and len(date_str) in [10, 13]:
            timestamp = int(date_str)
            if len(date_str) == 13:  # milliseconds
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp)
        # Handle float timestamps
        elif re.match(r'^\d+\.\d+$', date_str):
            timestamp = float(date_str)
            return datetime.fromtimestamp(timestamp)
    except:
        pass
    
    # Last resort: try fuzzy parsing with dateutil
    try:
        return dateutil.parser.parse(date_str, fuzzy=True)
    except:
        print(f"⚠️ Could not parse date: {date_str}")
        return None

# ------------------ DATE RANGE VALIDATION ------------------
def validate_date_range_data(df, date_col, start_date, end_date):
    """Validate if the date range contains any data"""
    if not date_col or date_col not in df.columns:
        return True  # No date column, so we can't validate
        
    # Parse dates using robust parser
    df_copy = df.copy()
    df_copy[date_col] = df_copy[date_col].apply(robust_date_parser)
    df_copy = df_copy[df_copy[date_col].notna()]
    
    if len(df_copy) == 0:
        raise HTTPException(status_code=400, detail="No valid date data found in the dataset")
    
    # Parse start and end dates
    start_dt = robust_date_parser(start_date) if isinstance(start_date, str) else start_date
    end_dt = robust_date_parser(end_date) if isinstance(end_date, str) else end_date
    
    if start_dt is None or end_dt is None:
        raise HTTPException(status_code=400, detail="Invalid start_date or end_date format")
    
    # Check if any data exists in the date range
    date_mask = (df_copy[date_col] >= start_dt) & (df_copy[date_col] <= end_dt)
    data_in_range = df_copy[date_mask]
    
    if len(data_in_range) == 0:
        raise HTTPException(status_code=400, detail="No data available for the selected date range")
    
    return True

def preprocess_datetime_column(series):
    """Enhanced datetime preprocessing that handles multiple formats"""
    print(f"🔄 Preprocessing datetime column with {len(series)} values")
    
    parsed_dates = []
    valid_indices = []
    
    for idx, date_val in series.items():
        parsed_date = robust_date_parser(date_val)
        if parsed_date is not None:
            parsed_dates.append(parsed_date)
            valid_indices.append(idx)
    
    if not parsed_dates:
        print("⚠️ Could not parse any dates, returning zeros")
        return pd.Series([0] * len(series), index=series.index)
    
    # Calculate days from reference date (most recent date in series)
    reference_date = max(parsed_dates)
    days_from_ref = [(reference_date - d).days for d in parsed_dates]
    
    # Create result series with same index as input
    result = pd.Series([0] * len(series), index=series.index)
    for i, idx in enumerate(valid_indices):
        result.loc[idx] = days_from_ref[i]
    
    print(f"✅ Datetime preprocessing completed: {len(parsed_dates)} valid dates out of {len(series)}")
    print(f"📅 Date range: {min(parsed_dates).strftime('%Y-%m-%d')} to {reference_date.strftime('%Y-%m-%d')}")
    
    return result

# ------------------ PREPROCESSING HELPERS ------------------
def detect_column_type(series):
    """Detect the type of data in a column"""
    # Check if it's numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    
    # Check if it's datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Try to parse as datetime using our robust parser
    sample_size = min(100, len(series))
    sample = series.dropna().head(sample_size)
    
    if len(sample) == 0:
        return "text"
    
    date_count = 0
    for val in sample:
        parsed = robust_date_parser(val)
        if parsed is not None:
            date_count += 1
    
    if date_count / sample_size > 0.7:  # If 70%+ can be parsed as dates
        return "datetime_string"
    
    # Check for categorical with limited unique values
    unique_ratio = len(sample.unique()) / len(sample)
    if unique_ratio < 0.3:  # If less than 30% unique values
        return "categorical"
    
    # Check for specific categorical patterns
    common_patterns = {
        'gender': ['M', 'F', 'Male', 'Female', 'MALE', 'FEMALE', 'm', 'f'],
        'boolean': ['Y', 'N', 'Yes', 'No', 'YES', 'NO', '1', '0', 'True', 'False']
    }
    
    for pattern_name, patterns in common_patterns.items():
        pattern_matches = sum(1 for val in sample if str(val).strip() in patterns)
        if pattern_matches / sample_size > 0.8:  # 80% match known patterns
            return f"categorical_{pattern_name}"
    
    return "text"

def preprocess_categorical_column(series, specific_type=None):
    """Preprocess categorical columns using appropriate encoding"""
    print(f"🔄 Preprocessing categorical column: {specific_type if specific_type else 'general'}")
    
    # Clean the data
    cleaned_series = series.fillna('Unknown').astype(str).str.strip().str.upper()
    
    if specific_type == 'categorical_gender':
        # Standardize gender values
        gender_mapping = {
            'M': 0, 'MALE': 0, 'MALES': 0, '1': 0,
            'F': 1, 'FEMALE': 1, 'FEMALES': 1, '2': 1,
            'UNKNOWN': 0.5, 'OTHER': 0.5, '': 0.5
        }
        result = cleaned_series.map(lambda x: gender_mapping.get(x, 0.5))
        print(f"✅ Gender encoding: {result.value_counts().to_dict()}")
        
    elif specific_type == 'categorical_boolean':
        # Standardize boolean values
        boolean_mapping = {
            'Y': 1, 'YES': 1, 'TRUE': 1, '1': 1, 'T': 1,
            'N': 0, 'NO': 0, 'FALSE': 0, '0': 0, 'F': 0,
            'UNKNOWN': 0.5, '': 0.5
        }
        result = cleaned_series.map(lambda x: boolean_mapping.get(x, 0.5))
        print(f"✅ Boolean encoding: {result.value_counts().to_dict()}")
        
    else:
        # General categorical encoding - use label encoding for limited categories, 
        # frequency encoding for high cardinality
        unique_count = len(cleaned_series.unique())
        
        if unique_count <= 20:  # Low cardinality - use label encoding
            le = LabelEncoder()
            result = pd.Series(le.fit_transform(cleaned_series), index=series.index)
            print(f"✅ Label encoding applied for {unique_count} categories")
        else:  # High cardinality - use frequency encoding
            freq_encoding = cleaned_series.value_counts(normalize=True)
            result = cleaned_series.map(freq_encoding)
            print(f"✅ Frequency encoding applied for {unique_count} categories")
    
    return result

def preprocess_numeric_column(series):
    """Preprocess numeric columns - handle outliers and missing values"""
    print(f"🔄 Preprocessing numeric column with {len(series)} values")
    
    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Fill NaN with median
    if numeric_series.isna().any():
        median_val = numeric_series.median()
        numeric_series = numeric_series.fillna(median_val)
        print(f"✅ Filled {numeric_series.isna().sum()} NaN values with median: {median_val}")
    
    # Handle outliers using IQR method
    Q1 = numeric_series.quantile(0.25)
    Q3 = numeric_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
    if outliers_mask.any():
        # Cap outliers instead of removing
        numeric_series = numeric_series.clip(lower_bound, upper_bound)
        print(f"✅ Capped {outliers_mask.sum()} outliers using IQR method")
    
    print(f"✅ Numeric preprocessing completed: range [{numeric_series.min():.2f}, {numeric_series.max():.2f}]")
    return numeric_series

def preprocess_additional_attributes(df, additional_attributes):
    """Preprocess all additional attributes for RFM++ segmentation"""
    print(f"🔄 Starting preprocessing for {len(additional_attributes)} additional attributes")
    
    preprocessing_results = {
        'successful_attributes': [],
        'failed_attributes': [],
        'preprocessed_data': {}
    }
    
    for attr in additional_attributes:
        attr_id = attr.id
        if attr_id not in df.columns:
            print(f"❌ Attribute '{attr_id}' not found in dataset columns")
            preprocessing_results['failed_attributes'].append(attr_id)
            continue
        
        try:
            print(f"🔧 Processing attribute: {attr_id}")
            series = df[attr_id]
            col_type = detect_column_type(series)
            print(f"   Detected type: {col_type}")
            
            if col_type == "numeric":
                processed_series = preprocess_numeric_column(series)
                
            elif col_type in ["datetime", "datetime_string"]:
                processed_series = preprocess_datetime_column(series)
                
            elif col_type.startswith("categorical"):
                processed_series = preprocess_categorical_column(series, col_type)
                
            else:  # text or unknown
                print(f"⚠️ Unsupported type '{col_type}' for {attr_id}, attempting categorical encoding")
                processed_series = preprocess_categorical_column(series)
            
            # Validate preprocessing result
            if processed_series is not None and len(processed_series) == len(df):
                preprocessing_results['preprocessed_data'][attr_id] = processed_series
                preprocessing_results['successful_attributes'].append(attr_id)
                print(f"✅ Successfully preprocessed {attr_id}")
            else:
                raise ValueError("Preprocessing resulted in invalid series")
                
        except Exception as e:
            print(f"❌ Failed to preprocess {attr_id}: {str(e)}")
            preprocessing_results['failed_attributes'].append(attr_id)
    
    print(f"📊 Preprocessing summary: {len(preprocessing_results['successful_attributes'])} successful, "
          f"{len(preprocessing_results['failed_attributes'])} failed")
    
    return preprocessing_results

# ------------------ UPDATED HELPERS WITH FIXES ------------------
def load_dataset(selected_file_name):
    if selected_file_name not in DATASETS:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    df = pd.read_csv(DATASETS[selected_file_name])
    return df

def filter_by_date(df, date_col, start_date, end_date):
    # If no date column provided, return full dataset
    if not date_col or date_col not in df.columns:
        print("⚠️ No valid date column provided, using full dataset")
        return df
    
    # Enhanced date parsing with robust parser
    print(f"🔄 Parsing date column '{date_col}' with robust parser...")
    df[date_col] = df[date_col].apply(robust_date_parser)
    
    # Remove rows where date parsing failed
    original_count = len(df)
    df = df[df[date_col].notna()].copy()
    if len(df) < original_count:
        print(f"⚠️ Removed {original_count - len(df)} rows with invalid dates")
    
    # Handle empty start_date or end_date
    if not start_date or start_date == "":
        start_date = df[date_col].min()
        print(f"⚠️ No start_date provided, using earliest date: {start_date}")
    else:
        start_date = robust_date_parser(start_date)
        if start_date is None:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
    
    if not end_date or end_date == "":
        end_date = df[date_col].max()
        print(f"⚠️ No end_date provided, using latest date: {end_date}")
    else:
        end_date = robust_date_parser(end_date)
        if end_date is None:
            raise HTTPException(status_code=400, detail="Invalid end_date format")
    
    # NEW: Check if date range has data before filtering
    date_mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    data_in_range = df[date_mask]
    
    if len(data_in_range) == 0:
        raise HTTPException(status_code=400, detail="No data available for the selected date range")
    
    # Filter by date range
    filtered_df = df.loc[date_mask].copy()
    print(f"✅ Date filter applied: {len(filtered_df)} transactions between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
    
    return filtered_df

def build_rfm(df, req: SegmentRequest, start_date, end_date):
    uid = req.uid_col
    recency_col = req.recency_col
    freq_col = req.frequency_col if req.frequency_col else None
    mon_col = req.monetary_col
    
    # Handle end_date - if empty, use max date from dataset
    if not end_date or end_date == "":
        df[recency_col] = df[recency_col].apply(robust_date_parser)
        end_date = df[recency_col].max()
        print(f"🔄 No end_date provided, using latest transaction date: {end_date}")
    
    today = robust_date_parser(end_date) if isinstance(end_date, str) else end_date

    print(f"Calculating RFM per user ({uid}) for date range {start_date} to {end_date}...")

    # ✅ Enhanced date parsing for recency column
    df[recency_col] = df[recency_col].apply(robust_date_parser)
    
    # Remove rows where recency date parsing failed
    original_count = len(df)
    df = df[df[recency_col].notna()].copy()
    if len(df) < original_count:
        print(f"⚠️ Removed {original_count - len(df)} rows with invalid recency dates")
    
    # Filter out transactions outside the date range for RFM calculation
    if start_date and end_date:
        start_dt = robust_date_parser(start_date) if isinstance(start_date, str) else start_date
        end_dt = robust_date_parser(end_date) if isinstance(end_date, str) else end_date
        
        date_filtered_df = df[
            (df[recency_col] >= start_dt) & 
            (df[recency_col] <= end_dt)
        ].copy()
        print(f"📊 Transactions in date range for RFM: {len(date_filtered_df)}")
    else:
        date_filtered_df = df.copy()
        print(f"📊 Using full dataset for RFM: {len(date_filtered_df)} transactions")

    # Group by UID
    grouped = date_filtered_df.groupby(uid)

    rfm = pd.DataFrame({
        "recency": grouped[recency_col].max().apply(lambda x: (today - x).days),
        "monetary": grouped[mon_col].sum()
    })

    # Frequency count
    if freq_col and freq_col in df.columns:
        rfm["frequency"] = grouped[freq_col].count()
    else:
        rfm["frequency"] = grouped[mon_col].count()

    rfm.reset_index(inplace=True)
    rfm.fillna(0, inplace=True)

    print(f"✅ RFM table created for {len(rfm)} unique users in the date range.")
    return rfm

def build_rfm_plus_plus(df, req: SegmentRequest, start_date, end_date, preprocessing_results=None):
    """Build RFM++ with preprocessed additional attributes"""
    print("🔄 Building RFM++ with additional attributes...")
    
    # First build basic RFM
    rfm = build_rfm(df, req, start_date, end_date)
    
    # Add preprocessed additional attributes
    if preprocessing_results and preprocessing_results['preprocessed_data']:
        additional_data_list = []
        
        for attr_id, processed_series in preprocessing_results['preprocessed_data'].items():
            # Group by user ID (take mean for numeric, mode for categorical-like)
            uid = req.uid_col
            if processed_series.dtype in ['int64', 'float64']:
                attr_data = df.groupby(uid).apply(
                    lambda x: processed_series.loc[x.index].mean() if not processed_series.loc[x.index].empty else 0
                )
            else:
                attr_data = df.groupby(uid).apply(
                    lambda x: processed_series.loc[x.index].mode().iloc[0] if not processed_series.loc[x.index].mode().empty else 'Unknown'
                )
            
            additional_data_list.append(attr_data.rename(attr_id))
        
        if additional_data_list:
            # Combine all additional attributes
            additional_df = pd.concat(additional_data_list, axis=1)
            
            # Merge with RFM
            rfm_plus = rfm.merge(additional_df, left_on=uid, right_index=True, how='left')
            rfm_plus.fillna(0, inplace=True)
            
            print(f"✅ RFM++ table created with {len(additional_df.columns)} additional attributes")
            return rfm_plus
    
    print("⚠️ No valid additional attributes found, using basic RFM")
    return rfm

def normalize_weights(weights):
    """Normalize weights to sum to 100%"""
    if weights is None:
        return None
        
    # Extract weights, defaulting to equal weights if not provided
    r_weight = weights.recency_weight if weights.recency_weight is not None else 33.33
    f_weight = weights.frequency_weight if weights.frequency_weight is not None else 33.33
    m_weight = weights.monetary_weight if weights.monetary_weight is not None else 33.34
    
    # Normalize to sum to 100
    total = r_weight + f_weight + m_weight
    if total == 0:
        return None
        
    normalized_r = (r_weight / total) * 100
    normalized_f = (f_weight / total) * 100
    normalized_m = (m_weight / total) * 100
    
    print(f"📊 Original weights - R: {r_weight}, F: {f_weight}, M: {m_weight}")
    print(f"📊 Normalized weights - R: {normalized_r:.2f}%, F: {normalized_f:.2f}%, M: {normalized_m:.2f}%")
    
    return {
        'recency': normalized_r / 100,  # Convert to decimal
        'frequency': normalized_f / 100,
        'monetary': normalized_m / 100
    }

def apply_weighted_scaling(features_df, feature_columns, weights):
    """Apply weighted scaling to features based on provided weights"""
    print(f"⚖️ Applying weighted scaling to {len(feature_columns)} features...")
    
    # Separate RFM features from additional attributes
    rfm_features = ['recency', 'frequency', 'monetary']
    additional_features = [f for f in feature_columns if f not in rfm_features]
    
    # Scale all features first
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df[feature_columns])
    scaled_df = pd.DataFrame(scaled_features, columns=feature_columns, index=features_df.index)
    
    # Apply weights to RFM features
    if weights:
        for feature, weight in weights.items():
            if feature in scaled_df.columns:
                scaled_df[feature] = scaled_df[feature] * weight
                print(f"   Applied weight {weight:.3f} to {feature}")
    
    print("✅ Weighted scaling completed.")
    return scaled_df.values

def scale_features(features_df, feature_columns, weights=None):
    """Scale features with optional weighting for RFM features"""
    if weights:
        return apply_weighted_scaling(features_df, feature_columns, weights)
    else:
        print(f"Scaling {len(feature_columns)} features (no weights)...")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features_df[feature_columns])
        print("✅ Scaling completed.")
        return scaled

def select_best_k(scaled_features, max_k=10):
    """
    Automatically selects best K using:
      1. Elbow method (KneeLocator)
      2. Silhouette score
      3. Calinski–Harabasz index
    Returns the most consistent best K value.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from kneed import KneeLocator

    wcss = []
    silhouette_scores = []
    ch_scores = []
    K_values = range(2, max_k + 1)

    # Calculate metrics for each K
    for k in K_values:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(scaled_features)
            wcss.append(km.inertia_)

            sil_score = silhouette_score(scaled_features, labels)
            ch_score = calinski_harabasz_score(scaled_features, labels)

            silhouette_scores.append(sil_score)
            ch_scores.append(ch_score)
        except Exception as e:
            print(f"⚠️ Skipping k={k}: {e}")
            wcss.append(np.nan)
            silhouette_scores.append(np.nan)
            ch_scores.append(np.nan)

    # --- Elbow Method (KneeLocator)
    try:
        kl = KneeLocator(K_values, wcss, curve="convex", direction="decreasing")
        elbow_k = kl.elbow if kl.elbow is not None else np.nan
    except Exception:
        elbow_k = np.nan

    # --- Best by Silhouette
    silhouette_k = K_values[int(np.nanargmax(silhouette_scores))]

    # --- Best by Calinski–Harabasz
    ch_k = K_values[int(np.nanargmax(ch_scores))]

    # --- Combine and pick the most frequent/best median K
    valid_ks = [k for k in [elbow_k, silhouette_k, ch_k] if not np.isnan(k)]
    if not valid_ks:
        print("⚠️ Could not determine best K, defaulting to 5.")
        return 5

    # Take the most common / median K among methods
    best_k = int(np.median(valid_ks))
    
    # 👈 NEW: Business requirement - if detected K is less than 5, add 3
    if best_k < 5:
        print(f"📈 Business adjustment: Detected K={best_k} is less than 5, adding 3 for better segmentation")
        best_k += 3
        print(f"📈 Adjusted K value: {best_k}")

    print("\n📊 Auto K-Selection Summary:")
    print(f"  Elbow (KneeLocator): {elbow_k}")
    print(f"  Silhouette Score Best: {silhouette_k}")
    print(f"  Calinski–Harabasz Best: {ch_k}")
    print(f"✅ Final Best K (Auto Consensus) = {best_k}\n")

    return best_k

def get_final_k_value(req: SegmentRequest, scaled_features):
    """
    Determine final K value based on user input or auto-selection
    """
    # 👈 NEW: If user provided K value, use it directly
    if req.k_value is not None and req.k_value > 0:
        print(f"🎯 Using user-provided K value: {req.k_value}")
        return req.k_value
    
    # Otherwise, use auto-selection with business adjustment
    print("🔍 No user K value provided, using auto-selection...")
    return select_best_k(scaled_features)

# ------------------ UPDATED SUMMARIZE AND AI FUNCTIONS ------------------
def summarize_segment(seg_df, is_advanced=False):
    """Summarize segment with optional additional attributes - WITH ROUNDING FIX"""
    # Round values to nearest whole number (ceiling for recency, frequency, monetary)
    summary = {
        "recency": {
            "avg": int(np.ceil(seg_df['recency'].mean())),
            "min": int(seg_df['recency'].min()),
            "max": int(seg_df['recency'].max())
        },
        "frequency": {
            "avg": int(np.ceil(seg_df['frequency'].mean())),
            "min": int(seg_df['frequency'].min()),
            "max": int(seg_df['frequency'].max())
        },
        "monetary": {
            "avg": int(np.ceil(seg_df['monetary'].mean())),
            "min": int(seg_df['monetary'].min()),
            "max": int(seg_df['monetary'].max())
        }
    }
    
    if is_advanced:
        # Include additional attributes in summary
        additional_cols = [col for col in seg_df.columns if col not in ['recency', 'frequency', 'monetary', 'segment_id']]
        for col in additional_cols:
            if seg_df[col].dtype in ['int64', 'float64']:
                summary[col] = {
                    "avg": float(seg_df[col].mean()),
                    "min": float(seg_df[col].min()),
                    "max": float(seg_df[col].max())
                }
            else:
                # For categorical, show value counts
                top_values = seg_df[col].value_counts().head(3).to_dict()
                summary[col] = {"top_values": top_values}
    
    return summary

def generate_segment_info(summary, segment_id, total_segments, all_summaries=None, is_advanced=False):
    """Generate AI-driven segment insights - WITH ROUNDING FIX"""
    
    # Build comparison context
    comparison_context = ""
    if all_summaries:
        comparison_context_list = []
        for idx, s in enumerate(all_summaries):
            if idx == segment_id:
                continue
            comparison_context_list.append(
                f"Segment {idx+1}: Recency avg={s['recency']['avg']:.0f}, "
                f"Frequency avg={s['frequency']['avg']:.0f}, Monetary avg={s['monetary']['avg']:.0f}"
            )
        if comparison_context_list:
            comparison_context = "\n- " + "\n- ".join(comparison_context_list)

    # Build additional attributes context for advanced segmentation
    additional_context = ""
    if is_advanced:
        additional_cols = [col for col in summary.keys() if col not in ['recency', 'frequency', 'monetary']]
        if additional_cols:
            additional_context = "\n\nAdditional Attributes:"
            for col in additional_cols:
                if 'avg' in summary[col]:
                    additional_context += f"\n- {col}: avg={summary[col]['avg']:.1f} (range: {summary[col]['min']}-{summary[col]['max']})"
                elif 'top_values' in summary[col]:
                    top_vals = ", ".join([f"{k}({v})" for k, v in summary[col]['top_values'].items()])
                    additional_context += f"\n- {col}: top values = {top_vals}"

    prompt = f"""
You are an AI assistant for customer segmentation.

I have {total_segments} customer segments. This is segment {segment_id + 1} of {total_segments}.

RFM summary for this segment:
- Recency: avg={summary['recency']['avg']:.0f} days (range: {summary['recency']['min']}-{summary['recency']['max']})
- Frequency: avg={summary['frequency']['avg']:.0f} transactions (range: {summary['frequency']['min']}-{summary['frequency']['max']})
- Monetary: avg={summary['monetary']['avg']:.0f} (range: {summary['monetary']['min']}-{summary['monetary']['max']})
{additional_context}

Other segments for comparison:{comparison_context}

Generate a JSON with:
- "segment_name": UNIQUE, creative 2–3 word name that reflects this segment's RFM pattern
- "segment_information": brief description specific to this segment's characteristics
- "characteristics": list 3–5 short traits differentiating this segment
- "segmentation_findings": 1–2 sentences analyzing behavior and tendencies
- "recommendations": list 3–5 actionable steps for this segment

Return ONLY valid JSON.
"""

    try:
        if not openai.api_key:
            raise Exception("OpenAI API key not set")

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=30
        )
        content = response['choices'][0]['message']['content']

        # Extract JSON only
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")

        content_json = content[start:end]
        ai_info = json.loads(content_json)

        # Validate required fields
        required_fields = ["segment_name", "segment_information", "characteristics", "segmentation_findings", "recommendations"]
        for field in required_fields:
            if field not in ai_info:
                raise ValueError(f"Missing required field: {field}")

        # Ensure lists
        if not isinstance(ai_info.get("characteristics"), list):
            ai_info["characteristics"] = list(ai_info["characteristics"])
        if not isinstance(ai_info.get("recommendations"), list):
            ai_info["recommendations"] = list(ai_info["recommendations"])

        return ai_info

    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return create_fallback_segment_info(summary, is_advanced)

def create_fallback_segment_info(summary, is_advanced=False):
    """Create meaningful fallback segment info without AI - WITH ROUNDING FIX"""
    recency_avg = summary['recency']['avg']
    frequency_avg = summary['frequency']['avg']
    monetary_avg = summary['monetary']['avg']
    
    # Determine segment type based on RFM values
    if recency_avg < 30 and frequency_avg > 50 and monetary_avg > 500000:
        segment_type = "High Value Loyal"
    elif recency_avg < 30 and frequency_avg > 20:
        segment_type = "Active Regulars" 
    elif recency_avg > 90:
        segment_type = "At Risk"
    elif frequency_avg < 5:
        segment_type = "New/Light"
    else:
        segment_type = "Medium Value"
    
    info = {
        "segment_name": segment_type,
        "segment_information": f"This segment shows average recency of {recency_avg:.0f} days, frequency of {frequency_avg:.0f} transactions, and monetary value of {monetary_avg:.0f}",
        "characteristics": {
            "recency": f"Customers have average recency of {recency_avg:.0f} days since last purchase",
            "frequency": f"Average purchase frequency is {frequency_avg:.0f} transactions", 
            "monetary": f"Average spending per customer is {monetary_avg:.0f}"
        },
        "segmentation_findings": {
            "recency": f"Recency ranges from {summary['recency']['min']} to {summary['recency']['max']} days",
            "frequency": f"Frequency ranges from {summary['frequency']['min']} to {summary['frequency']['max']} transactions",
            "monetary": f"Monetary value ranges from {summary['monetary']['min']} to {summary['monetary']['max']}"
        },
        "recommendations": {
            "recency": "Consider engagement campaigns based on recency patterns",
            "frequency": "Develop frequency-based loyalty programs", 
            "monetary": "Create targeted offers to increase average spend"
        }
    }
    
    if is_advanced:
        info["segment_information"] += " (Enhanced with additional attributes)"
    
    return info

# ------------------ UPDATED SEGMENTATION FUNCTIONS ------------------
def perform_basic_segmentation(df_filtered, req, start_date, end_date):
    """Perform basic RFM segmentation with optional weights and K value"""
    print("🔹 Performing BASIC RFM segmentation...")
    
    # Build RFM
    rfm = build_rfm(df_filtered, req, start_date, end_date)
    total_customers = len(rfm)
    
    # Normalize weights if provided
    weights = normalize_weights(req.rfm_weights) if req.rfm_weights else None
    if weights:
        print(f"⚖️ Using weighted RFM features: R={weights['recency']:.3f}, F={weights['frequency']:.3f}, M={weights['monetary']:.3f}")
    
    # Scale RFM features with optional weights
    scaled_features = scale_features(rfm, ["recency", "frequency", "monetary"], weights)
    
    # 👈 NEW: Get final K value (user input or auto-selected)
    best_k = get_final_k_value(req, scaled_features)
    
    # Cluster
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm['segment_id'] = kmeans.fit_predict(scaled_features)
    
    return rfm, total_customers, best_k, False, weights

def perform_advanced_segmentation(df_filtered, req, start_date, end_date):
    """Perform advanced RFM++ segmentation with comprehensive preprocessing and optional weights"""
    print("🔹 Attempting ADVANCED RFM++ segmentation with preprocessing...")
    
    try:
        # Step 1: Preprocess additional attributes
        preprocessing_results = preprocess_additional_attributes(
            df_filtered, 
            req.additional_attributes
        )
        
        # Check if we have enough successfully preprocessed attributes
        successful_count = len(preprocessing_results['successful_attributes'])
        total_attributes = len(req.additional_attributes)
        
        print(f"📊 Preprocessing results: {successful_count}/{total_attributes} attributes successful")
        
        if successful_count == 0:
            print("❌ No attributes could be preprocessed successfully, falling back to basic RFM")
            return perform_basic_segmentation(df_filtered, req, start_date, end_date)
        
        if successful_count < total_attributes:
            print(f"⚠️ Only {successful_count}/{total_attributes} attributes available for RFM++")
        
        # Step 2: Build RFM++ with preprocessed attributes
        rfm_plus = build_rfm_plus_plus(
            df_filtered, req, start_date, end_date, preprocessing_results
        )
        total_customers = len(rfm_plus)
        
        # Step 3: Identify feature columns (RFM + successfully preprocessed additional attributes)
        feature_columns = ["recency", "frequency", "monetary"]
        successful_cols = preprocessing_results['successful_attributes']
        feature_columns.extend(successful_cols)
        
        print(f"🔄 Using {len(feature_columns)} features for clustering: {feature_columns}")
        
        # Step 4: Normalize weights if provided (only applied to RFM features)
        weights = normalize_weights(req.rfm_weights) if req.rfm_weights else None
        if weights:
            print(f"⚖️ Using weighted RFM features: R={weights['recency']:.3f}, F={weights['frequency']:.3f}, M={weights['monetary']:.3f}")
        
        # Step 5: Scale all features with optional weights for RFM features
        scaled_features = scale_features(rfm_plus, feature_columns, weights)
        
        # 👈 NEW: Get final K value (user input or auto-selected)
        best_k = get_final_k_value(req, scaled_features)
        
        # Step 6: Cluster
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        rfm_plus['segment_id'] = kmeans.fit_predict(scaled_features)
        
        return rfm_plus, total_customers, best_k, True, weights
        
    except Exception as e:
        print(f"❌ Advanced segmentation failed: {e}")
        print("🔄 Falling back to basic RFM segmentation...")
        return perform_basic_segmentation(df_filtered, req, start_date, end_date)

# ------------------ ROUTES AND BACKGROUND TASKS (UPDATED) ------------------
job_results = {}

# Background task to process RFM job/ auto segmentation
def process_rfm_job(job_id: str, req: SegmentRequest):
    print("==== RFM Segmentation Summary API Started ====")
    print(f"Request: {req.dict()}")
    print(f"Additional attributes provided: {len(req.additional_attributes) if req.additional_attributes else 0}")
    print(f"RFM Weights provided: {req.rfm_weights.dict() if req.rfm_weights else 'None'}")
    print(f"K Value provided: {req.k_value if req.k_value else 'Auto-selection'}")

    try:
        job_results[job_id] = {"status": "PROCESSING"}

        # Load dataset
        df = load_dataset(req.selected_file_name)

        # 🔹 Get UID column dynamically from request
        uid_col = req.uid_col or "customer_id"
        if uid_col not in df.columns:
            raise ValueError(f"UID column '{uid_col}' not found in dataset")

        # Auto-select date_col
        if not req.date_col:
            if req.selected_file_name == "Dataset1":
                date_col = "date_of_use"
            elif req.selected_file_name == "Dataset2":
                date_col = "created_at"
            else:
                date_col = None
        else:
            date_col = req.date_col

        # NEW: Early validation for date range data
        print("🔍 Validating date range data...")
        validate_date_range_data(df, date_col, req.start_date, req.end_date)

        # Filter - WITH UPDATED DATE HANDLING
        df_filtered = filter_by_date(df, date_col, req.start_date, req.end_date)

        # Choose segmentation type
        if req.additional_attributes and len(req.additional_attributes) > 0:
            # Attempt advanced segmentation with preprocessing and fallback
            rfm_result, total_customers, best_k, is_advanced, weights_used = perform_advanced_segmentation(
                df_filtered, req, req.start_date, req.end_date
            )
            segmentation_type = "ADVANCED_RFM++" if is_advanced else "BASIC_RFM (fallback)"
        else:
            # Basic segmentation
            rfm_result, total_customers, best_k, is_advanced, weights_used = perform_basic_segmentation(
                df_filtered, req, req.start_date, req.end_date
            )
            segmentation_type = "BASIC_RFM"

        print(f"✅ Segmentation completed: {segmentation_type}, K={best_k}")

        # Prepare all summaries first
        all_summaries = []
        for seg_id in sorted(rfm_result['segment_id'].unique()):
            seg_df = rfm_result[rfm_result['segment_id'] == seg_id]
            summary = summarize_segment(seg_df, is_advanced)
            all_summaries.append(summary)

        # Generate detailed info for each segment
        segments = []
        user_mapping = []  # ✅ for uid mapping
        total_segments = len(all_summaries)

        for seg_id, summary in enumerate(all_summaries):
            seg_df = rfm_result[rfm_result['segment_id'] == seg_id]

            # AI info with context
            ai_info = generate_segment_info(
                summary,
                segment_id=seg_id,
                total_segments=total_segments,
                all_summaries=all_summaries,
                is_advanced=is_advanced
            )

            # Retry if needed
            if not ai_info.get("segment_name") or ai_info["segment_name"].strip() == "":
                ai_info = generate_segment_info(summary, seg_id, total_segments, all_summaries, is_advanced)

            segment_name = ai_info.get("segment_name", f"Segment {seg_id}")

            segment_data = {
                "segment_id": f"segment_{seg_id}",
                "segment_name": ai_info.get("segment_name", f"Segment {seg_id}"),
                "customer_count": len(seg_df),
                "percent": round(len(seg_df) / len(rfm_result) * 100, 2),
                "summary_stats": {
                    "recency_days_avg": summary['recency']['avg'],
                    "recency_days_min": summary['recency']['min'],
                    "recency_days_max": summary['recency']['max'],
                    "frequency_days_avg": summary['frequency']['avg'],
                    "frequency_days_min": summary['frequency']['min'],
                    "frequency_days_max": summary['frequency']['max'],
                    "monetary_avg": summary['monetary']['avg'],
                    "monetary_min": summary['monetary']['min'],
                    "monetary_max": summary['monetary']['max']
                },
                "segment_information": ai_info.get("segment_information", ""),
                "characteristics": ai_info.get("characteristics", {}),
                "segmentation_findings": ai_info.get("segmentation_findings", ""),
                "recommendations": ai_info.get("recommendations", {})
            }

            segments.append(segment_data)

            # ✅ Add UID list for this segment
            uid_list = seg_df[uid_col].dropna().astype(str).unique().tolist()
            user_mapping.append({
                "segment_id": f"segment_{seg_id}",
                "segment_name": ai_info.get("segment_name", f"Segment {seg_id}"),
                "user_count": len(uid_list),   # ✅ added user count
                "uid_list": uid_list
            })

        # Prepare weights info for response
        weights_info = None
        if weights_used:
            weights_info = {
                "recency_weight_percentage": round(weights_used['recency'] * 100, 2),
                "frequency_weight_percentage": round(weights_used['frequency'] * 100, 2),
                "monetary_weight_percentage": round(weights_used['monetary'] * 100, 2)
            }

        # Mark COMPLETED
        job_results[job_id] = {
            "status": "COMPLETED",
            "dataset_name": req.selected_file_name,
            "date_range": f"{req.start_date} to {req.end_date}",
            "total_customers": total_customers,
            "segmentation_type": segmentation_type,
            "k_value": best_k,
            "k_value_source": "user_provided" if req.k_value else "auto_selected",
            "additional_attributes_used": is_advanced and req.additional_attributes is not None,
            "rfm_weights_used": weights_info,
            "segments": segments,
            "user_mapping": user_mapping
        }
        print("==== RFM Segmentation Summary API Completed ====")

    except Exception as e:
        job_results[job_id] = {"status": "FAILED", "error": str(e)}
        print("❌ Job failed:", e)

# Random cluster segmentation
def process_rfm_random_cluster_job(job_id: str, req: SegmentRequest):
    print("==== RFM Segmentation (Random Cluster Count) Started ====")
    try:
        job_results[job_id] = {"status": "PROCESSING"}

        df = load_dataset(req.selected_file_name)
        date_col = req.date_col or ("date_of_use" if req.selected_file_name == "Dataset1" else "created_at")
        
        # NEW: Early validation for date range data
        print("🔍 Validating date range data...")
        validate_date_range_data(df, date_col, req.start_date, req.end_date)
        
        df_filtered = filter_by_date(df, date_col, req.start_date, req.end_date)

        # Choose segmentation type with preprocessing
        if req.additional_attributes and len(req.additional_attributes) > 0:
            rfm_result, total_customers, _, is_advanced, weights_used = perform_advanced_segmentation(
                df_filtered, req, req.start_date, req.end_date
            )
            segmentation_type = "ADVANCED_RFM++" if is_advanced else "BASIC_RFM (fallback)"
        else:
            rfm_result, total_customers, _, is_advanced, weights_used = perform_basic_segmentation(
                df_filtered, req, req.start_date, req.end_date
            )
            segmentation_type = "BASIC_RFM"

        # Randomly pick cluster count
        import random
        cluster_options = [7, 9, 11]
        k_value = random.choice(cluster_options)
        print(f"🔹 Randomly selected K = {k_value}")

        # Re-cluster with random K
        feature_columns = ["recency", "frequency", "monetary"]
        if is_advanced and req.additional_attributes:
            preprocessing_results = preprocess_additional_attributes(df_filtered, req.additional_attributes)
            successful_cols = preprocessing_results['successful_attributes']
            feature_columns.extend(successful_cols)
        
        scaled_features = scale_features(rfm_result, feature_columns, weights_used)
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        rfm_result['segment_id'] = kmeans.fit_predict(scaled_features)

        # Generate summaries
        all_summaries = []
        for seg_id in sorted(rfm_result['segment_id'].unique()):
            seg_df = rfm_result[rfm_result['segment_id'] == seg_id]
            all_summaries.append(summarize_segment(seg_df, is_advanced))

        total_segments = len(all_summaries)

        # Prepare segments + AI info
        segments = []
        user_mapping = []

         # ✅ Get dynamic UID column name from request
        uid_col = req.uid_col if hasattr(req, "uid_col") else None
        if not uid_col or uid_col not in df_filtered.columns:
            raise ValueError("UID column missing or invalid in dataset/request.")

        for seg_id, summary in enumerate(all_summaries):
            seg_df = rfm_result[rfm_result['segment_id'] == seg_id]
            ai_info = generate_segment_info(summary, seg_id, total_segments, all_summaries, is_advanced)

            segment_name = ai_info.get("segment_name", f"Segment {seg_id}")

            # === 🔥 NEW PART: build user mapping for each segment ===
            uid_list = seg_df[uid_col].dropna().astype(str).unique().tolist()
            user_mapping.append({
                "segment_id": f"segment_{seg_id}",
                "segment_name": segment_name,
                "user_count": len(uid_list),  # ✅ Added user_count
                "uid_list": uid_list
            })

            
            segment_data = {
                "segment_id": f"segment_{seg_id}",
                "segment_name": ai_info.get("segment_name", f"Segment {seg_id}"),
                "customer_count": len(seg_df),
                "percent": round(len(seg_df) / total_customers * 100, 2),
                "summary_stats": summary,
                "segment_information": ai_info.get("segment_information", ""),
                "characteristics": ai_info.get("characteristics", []),
                "segmentation_findings": ai_info.get("segmentation_findings", ""),
                "recommendations": ai_info.get("recommendations", [])
            }
            segments.append(segment_data)

        # Prepare weights info for response
        weights_info = None
        if weights_used:
            weights_info = {
                "recency_weight_percentage": round(weights_used['recency'] * 100, 2),
                "frequency_weight_percentage": round(weights_used['frequency'] * 100, 2),
                "monetary_weight_percentage": round(weights_used['monetary'] * 100, 2)
            }

        job_results[job_id] = {
            "status": "COMPLETED",
            "dataset_name": req.selected_file_name,
            "date_range": f"{req.start_date} to {req.end_date}",
            "k_value": k_value,
            "k_value_source": "random_selection",
            "total_customers": total_customers,
            "segmentation_type": segmentation_type,
            "additional_attributes_used": is_advanced and req.additional_attributes is not None,
            "rfm_weights_used": weights_info,
            "segments": segments,
            "user_mapping": user_mapping
        }

        print("==== RFM Segmentation (Random Cluster Count) Completed ====")

    except Exception as e:
        job_results[job_id] = {"status": "FAILED", "error": str(e)}
        print("❌ Job failed:", e)

# Fixed cluster segmentation
def process_rfm_fixed_8_job(job_id: str, req: SegmentRequest):
    print("==== RFM Segmentation (Fixed K=8) Started ====")
    try:
        job_results[job_id] = {"status": "PROCESSING"}

        df = load_dataset(req.selected_file_name)
        date_col = req.date_col or ("date_of_use" if req.selected_file_name == "Dataset1" else "created_at")
        
        # NEW: Early validation for date range data
        print("🔍 Validating date range data...")
        validate_date_range_data(df, date_col, req.start_date, req.end_date)
        
        df_filtered = filter_by_date(df, date_col, req.start_date, req.end_date)

        # Choose segmentation type with preprocessing
        if req.additional_attributes and len(req.additional_attributes) > 0:
            rfm_result, total_customers, _, is_advanced, weights_used = perform_advanced_segmentation(
                df_filtered, req, req.start_date, req.end_date
            )
            segmentation_type = "ADVANCED_RFM++" if is_advanced else "BASIC_RFM (fallback)"
        else:
            rfm_result, total_customers, _, is_advanced, weights_used = perform_basic_segmentation(
                df_filtered, req, req.start_date, req.end_date
            )
            segmentation_type = "BASIC_RFM"

        # Fixed K = 8
        k_value = 8
        print(f"🔹 Using fixed K = {k_value}")

        # Re-cluster with fixed K
        feature_columns = ["recency", "frequency", "monetary"]
        if is_advanced and req.additional_attributes:
            preprocessing_results = preprocess_additional_attributes(df_filtered, req.additional_attributes)
            successful_cols = preprocessing_results['successful_attributes']
            feature_columns.extend(successful_cols)
        
        scaled_features = scale_features(rfm_result, feature_columns, weights_used)
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        rfm_result['segment_id'] = kmeans.fit_predict(scaled_features)

        # Generate summaries
        all_summaries = []
        for seg_id in sorted(rfm_result['segment_id'].unique()):
            seg_df = rfm_result[rfm_result['segment_id'] == seg_id]
            all_summaries.append(summarize_segment(seg_df, is_advanced))

        total_segments = len(all_summaries)

        # Prepare segments + AI info
        segments = []
        user_mapping = []

        # ✅ Get dynamic UID column name from request
        uid_col = req.uid_col if hasattr(req, "uid_col") else None
        if not uid_col or uid_col not in df_filtered.columns:
            raise ValueError("UID column missing or invalid in dataset/request.")

        for seg_id, summary in enumerate(all_summaries):
            seg_df = rfm_result[rfm_result['segment_id'] == seg_id]
            ai_info = generate_segment_info(summary, seg_id, total_segments, all_summaries, is_advanced)

            segment_name = ai_info.get("segment_name", f"Segment {seg_id}")

            # === 🔥 NEW PART: build user mapping for each segment ===
            uid_list = seg_df[uid_col].dropna().astype(str).unique().tolist()
            user_mapping.append({
                "segment_id": f"segment_{seg_id}",
                "segment_name": segment_name,
                "user_count": len(uid_list),  # ✅ Added user_count
                "uid_list": uid_list
            })

            segment_data = {
                "segment_id": f"segment_{seg_id}",
                "segment_name": ai_info.get("segment_name", f"Segment {seg_id}"),
                "customer_count": len(seg_df),
                "percent": round(len(seg_df) / total_customers * 100, 2),
                "summary_stats": summary,
                "segment_information": ai_info.get("segment_information", ""),
                "characteristics": ai_info.get("characteristics", []),
                "segmentation_findings": ai_info.get("segmentation_findings", ""),
                "recommendations": ai_info.get("recommendations", [])
            }
            segments.append(segment_data)

        # Prepare weights info for response
        weights_info = None
        if weights_used:
            weights_info = {
                "recency_weight_percentage": round(weights_used['recency'] * 100, 2),
                "frequency_weight_percentage": round(weights_used['frequency'] * 100, 2),
                "monetary_weight_percentage": round(weights_used['monetary'] * 100, 2)
            }

        job_results[job_id] = {
            "status": "COMPLETED",
            "dataset_name": req.selected_file_name,
            "date_range": f"{req.start_date} to {req.end_date}",
            "k_value": k_value,
            "k_value_source": "fixed",
            "total_customers": total_customers,
            "segmentation_type": segmentation_type,
            "additional_attributes_used": is_advanced and req.additional_attributes is not None,
            "rfm_weights_used": weights_info,
            "segments": segments,
            "user_mapping": user_mapping
        }

        print("==== RFM Segmentation (Fixed K=8) Completed ====")

    except Exception as e:
        job_results[job_id] = {"status": "FAILED", "error": str(e)}
        print("❌ Job failed:", e)

# ------------------ EXISTING ROUTES ------------------
@app.get("/")
def read_root():
    return {
        "message": """RFM Segmentation API is running.
Available endpoints:
- /columns : Get all columns of the datasource
- /suggest_rfm_cols : Suggested columns for RFM & Additional fields
- /suggest_rfm_cols_hardcoded : Hardcoded suggestions for RFM fields (Only for LC Data)
- /segments_summary_with_info : Generate RFM segments with AI-generated insights (auto K/auto segment count)
- /segments_summary_with_info_random : Generate RFM segments with AI-generated insights (random K from 7,9,11)
- /segments_summary_with_info_fixed : Generate RFM segments with AI-generated insights (fixed K=8)
- /job_status/{job_id} : Check status of background segmentation job
- """
    }

@app.post("/columns")
def get_dataset_columns(request: DatasetRequest):
    dataset_name = request.dataset_name
    if dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    try:
        df = pd.read_csv(DATASETS[dataset_name])
        return {"dataset_name": dataset_name, "columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")

@app.post("/suggest_rfm_cols")
def suggest_rfm_cols_llm_nice(request: DatasetRequest):
    dataset_name = request.dataset_name
    if dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    
    try:
        df = pd.read_csv(DATASETS[dataset_name], nrows=100)
        col_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            sample_vals = df[col].dropna().astype(str).tolist()[:5]
            col_info.append({"column_name": col, "dtype": col_type, "sample_values": sample_vals})

        prompt = f"""
You are an AI assistant for RFM segmentation and RFM++ analysis.

Here is a dataset column information (name, type, sample values):
{json.dumps(col_info, indent=2)}

Suggest **2–3 possible columns** for each of the following categories:

**RFM Core Attributes:**
- UID (user id)
- Recency (R)
- Frequency (F) 
- Monetary (M)

**RFM++ Additional Attributes** (for enhanced segmentation):
- Customer demographics (age, gender, location, etc.)
- Behavioral patterns (product categories, transaction types, etc.)
- Engagement metrics (customer tenure, loyalty program status, etc.)
- Geographic information
- Product/service preferences
- Customer lifecycle stage indicators

Return **strictly valid JSON** like this, with human-readable names (capitalize and replace underscores with spaces):
{{
  "UID_attributes": [{{"id": "col1", "name": "Nice Name"}}],
  "recency_attributes": [{{"id": "col1", "name": "Nice Name"}}],
  "frequency_attributes": [{{"id": "col1", "name": "Nice Name"}}],
  "monetary_attributes": [{{"id": "col1", "name": "Nice Name"}}],
  "additional_attributes": [
    {{"id": "col1", "name": "Nice Name", "category": "demographic/behavioral/engagement/geographic/preference/lifecycle"}},
    {{"id": "col2", "name": "Nice Name", "category": "demographic/behavioral/engagement/geographic/preference/lifecycle"}}
  ]
}}

For additional_attributes, include 5-8 most relevant columns that could enhance RFM segmentation.
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        content = response['choices'][0]['message']['content']
        result = json.loads(content)
        return {"dataset_name": dataset_name, "llm_suggestions": result}

    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/suggest_rfm_cols_hardcoded")
def suggest_rfm_cols_hardcoded(request: DatasetRequest):
    response = {
        "dataset_name": request.dataset_name,
        "llm_suggestions": {
            "UID_attributes": [
                {"id": "entry_number", "name": "Entry Number"},
                {"id": "contract_number", "name": "Contract Number"}
            ],
            "recency_attributes": [
                {"id": "date_of_use", "name": "Date of Use"},
                {"id": "repayment_date", "name": "Repayment Date"}
            ],
            "frequency_attributes": [
                {"id": "date_of_use", "name": "Date of Use"},
                {"id": "purchase_amount", "name": "Purchase Amount"},
                {"id": "entry_number", "name": "Entry Number"}
            ],
            "monetary_attributes": [
                {"id": "purchase_amount", "name": "Purchase Amount"},
                {"id": "annual_income", "name": "Annual Income"}
            ],
            "additional_attributes": [
                {"id": "gender", "name": "Gender", "category": "demographic"},
                {"id": "dob", "name": "Date of Birth", "category": "demographic"},
                {"id": "postal_code", "name": "Postal Code", "category": "demographic"}
            ]
        }
    }
    return response

@app.post("/segments_summary_with_info")
def segments_summary_with_info(req: SegmentRequest, background_tasks: BackgroundTasks):
    """Auto K selection segmentation with preprocessing and optional weights"""
    job_id = str(uuid4())
    job_results[job_id] = {"status": "PROCESSING"}
    background_tasks.add_task(process_rfm_job, job_id, req)
    return {"job_id": job_id, "status": "PROCESSING"}

@app.post("/segments_summary_with_info_random")
def segments_summary_random_cluster(req: SegmentRequest, background_tasks: BackgroundTasks):
    """Random K selection segmentation with preprocessing and optional weights"""
    job_id = str(uuid4())
    job_results[job_id] = {"status": "PROCESSING"}
    background_tasks.add_task(process_rfm_random_cluster_job, job_id, req)
    return {"job_id": job_id, "status": "PROCESSING"}

@app.post("/segments_summary_with_info_fixed")
def segments_summary_fixed_8(req: SegmentRequest, background_tasks: BackgroundTasks):
    """Fixed K=8 segmentation with preprocessing and optional weights"""
    job_id = str(uuid4())
    job_results[job_id] = {"status": "PROCESSING"}
    background_tasks.add_task(process_rfm_fixed_8_job, job_id, req)
    return {"job_id": job_id, "status": "PROCESSING"}

@app.get("/job_status/{job_id}")
def get_job_status(job_id: str):
    """Frontend polls this endpoint until COMPLETED."""
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Invalid job ID")
    return job_results[job_id]