import os
import sys
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import re
import json
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

# --- BUILT-IN FEATURE EXTRACTION ---
def extract_features(df, encoders=None):
    """
    Built-in feature extraction function for AWS CloudTrail logs.
    This creates basic features that might be used in threat detection.
    """
    df_feat = df.copy()
    
    # Convert eventTime to datetime if it's a string
    if 'eventTime' in df_feat.columns:
        df_feat['eventTime'] = pd.to_datetime(df_feat['eventTime'], errors='coerce')
        
        # Extract time-based features
        df_feat['hour'] = df_feat['eventTime'].dt.hour
        df_feat['day_of_week'] = df_feat['eventTime'].dt.dayofweek
        df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)
        df_feat['is_night'] = ((df_feat['hour'] >= 22) | (df_feat['hour'] <= 6)).astype(int)
    else:
        # Default values if eventTime is missing
        df_feat['hour'] = 12
        df_feat['day_of_week'] = 1
        df_feat['is_weekend'] = 0
        df_feat['is_night'] = 0
    
    # IP-based features
    if 'sourceIPAddress' in df_feat.columns:
        # Check if IP is private
        def is_private_ip(ip):
            if pd.isna(ip) or not isinstance(ip, str):
                return 0
            private_patterns = [
                r'^10\.',
                r'^192\.168\.',
                r'^172\.(1[6-9]|2[0-9]|3[0-1])\.'
            ]
            return int(any(re.match(pattern, ip) for pattern in private_patterns))
        
        df_feat['is_private_ip'] = df_feat['sourceIPAddress'].apply(is_private_ip)
        
        # New feature: is_external_ip
        df_feat['is_external_ip'] = (~df_feat['sourceIPAddress'].apply(lambda x: is_private_ip(x) == 1)).astype(int)

        # Extract IP octets (simplified)
        def get_first_octet(ip):
            if pd.isna(ip) or not isinstance(ip, str):
                return 0
            try:
                return int(ip.split('.')[0])
            except:
                return 0
        
        df_feat['ip_first_octet'] = df_feat['sourceIPAddress'].apply(get_first_octet)
    else:
        df_feat['is_private_ip'] = 0
        df_feat['is_external_ip'] = 0
        df_feat['ip_first_octet'] = 0
    
    # User Agent features
    if 'userAgent' in df_feat.columns:
        df_feat['user_agent_length'] = df_feat['userAgent'].fillna('').astype(str).str.len()
        df_feat['is_boto3'] = df_feat['userAgent'].fillna('').str.contains('boto3', case=False).astype(int)
        df_feat['is_console'] = df_feat['userAgent'].fillna('').str.contains('console', case=False).astype(int)
        df_feat['is_cli'] = df_feat['userAgent'].fillna('').str.contains('aws-cli', case=False).astype(int)
        # New feature: suspicious_user_agent
        suspicious_patterns = ['nmap', 'nessus', 'metasploit', 'sqlmap', 'python', 'curl', 'wget']
        df_feat['suspicious_user_agent'] = df_feat['userAgent'].fillna('').apply(lambda x: any(p in x.lower() for p in suspicious_patterns)).astype(int)
    else:
        df_feat['user_agent_length'] = 0
        df_feat['is_boto3'] = 0
        df_feat['is_console'] = 0
        df_feat['is_cli'] = 0
        df_feat['suspicious_user_agent'] = 0
    
    # Error-based features
    if 'errorCode' in df_feat.columns:
        df_feat['has_error'] = df_feat['errorCode'].notna().astype(int)
        df_feat['is_access_denied'] = df_feat['errorCode'].fillna('').str.contains('AccessDenied', case=False).astype(int)
    else:
        df_feat['has_error'] = 0
        df_feat['is_access_denied'] = 0
    
    # Event name features
    if 'eventName' in df_feat.columns:
        df_feat['is_create_event'] = df_feat['eventName'].fillna('').str.contains('Create', case=False).astype(int)
        df_feat['is_delete_event'] = df_feat['eventName'].fillna('').str.contains('Delete', case=False).astype(int)
        df_feat['is_list_event'] = df_feat['eventName'].fillna('').str.contains('List|Describe', case=False).astype(int)
        df_feat['is_admin_event'] = df_feat['eventName'].fillna('').str.contains('Admin|Root|Policy', case=False).astype(int)
        # New feature: is_sensitive_api
        sensitive_apis = ['CreateUser', 'AttachUserPolicy', 'CreateRole', 'AssumeRole', 
                         'GetSessionToken', 'PutBucketPolicy', 'ModifyDBInstance', 
                         'RunInstances', 'AuthorizeSecurityGroupIngress', 'CreateSnapshot']
        df_feat['is_sensitive_api'] = df_feat['eventName'].fillna('').apply(lambda x: x in sensitive_apis).astype(int)

        # Encoded eventName
        if encoders and 'eventName' in encoders:
            df_feat['eventName_encoded'] = df_feat['eventName'].apply(lambda x: encoders['eventName'].transform([x])[0] if x in encoders['eventName'].classes_ else -1)
        else:
            df_feat['eventName_encoded'] = 0 # Default if encoder not available
    else:
        df_feat['is_create_event'] = 0
        df_feat['is_delete_event'] = 0
        df_feat['is_list_event'] = 0
        df_feat['is_admin_event'] = 0
        df_feat['is_sensitive_api'] = 0
        df_feat['eventName_encoded'] = 0

    # Encoded eventSource
    if 'eventSource' in df_feat.columns and encoders and 'eventSource' in encoders:
        df_feat['eventSource_encoded'] = df_feat['eventSource'].apply(lambda x: encoders['eventSource'].transform([x])[0] if x in encoders['eventSource'].classes_ else -1)
    else:
        df_feat['eventSource_encoded'] = 0

    # Encoded awsRegion
    if 'awsRegion' in df_feat.columns and encoders and 'awsRegion' in encoders:
        df_feat['awsRegion_encoded'] = df_feat['awsRegion'].apply(lambda x: encoders['awsRegion'].transform([x])[0] if x in encoders['awsRegion'].classes_ else -1)
    else:
        df_feat['awsRegion_encoded'] = 0

    # Fill any remaining NaN values
    df_feat = df_feat.fillna(0)
    
    return df_feat

# --- SYNTHETIC DATA GENERATION ---
def generate_synthetic_cloudtrail_data(num_events=1000):
    """Generate realistic CloudTrail logs with attack scenarios"""
    
    # Normal baseline activities
    normal_events = []
    attack_events = []
    
    # Common AWS API calls for normal activities
    normal_apis = [
        'DescribeInstances', 'ListBuckets', 'GetObject', 'PutObject',
        'DescribeSecurityGroups', 'DescribeVpcs', 'GetUser', 'ListRoles'
    ]
    
    # Suspicious API calls for attacks
    attack_apis = [
        'CreateUser', 'AttachUserPolicy', 'CreateRole', 'AssumeRole',
        'GetSessionToken', 'PutBucketPolicy', 'ModifyDBInstance', 
        'RunInstances', 'AuthorizeSecurityGroupIngress', 'CreateSnapshot'
    ]
    
    # Generate normal events (70% of data)
    for i in range(int(num_events * 0.7)):
        event = {
            'eventTime': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
            'eventName': np.random.choice(normal_apis),
            'sourceIPAddress': f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
            'userAgent': 'aws-cli/2.0.0 Python/3.8.0',
            'errorCode': None if np.random.random() > 0.1 else 'AccessDenied',
            'responseElements': {'success': True} if np.random.random() > 0.1 else None,
            'requestParameters': {'region': 'us-east-1'},
            'recipientAccountId': '123456789012',
            'awsRegion': 'us-east-1',
            'eventSource': f"{np.random.choice(['ec2', 's3', 'iam', 'rds'])}.amazonaws.com",
            'threat_label': 0,  # Normal activity
            'risk_score': np.random.uniform(0, 30)  # Low risk
        }
        normal_events.append(event)
    
    # Generate attack scenarios (30% of data)
    attack_scenarios = [
        'privilege_escalation', 'credential_theft', 'data_exfiltration', 
        'reconnaissance', 'persistence', 'lateral_movement'
    ]
    
    for i in range(int(num_events * 0.3)):
        scenario = np.random.choice(attack_scenarios)
        
        # Privilege escalation pattern
        if scenario == 'privilege_escalation':
            event = {
                'eventTime': (datetime.now() - timedelta(hours=np.random.randint(1, 48))).isoformat(),
                'eventName': np.random.choice(['CreateUser', 'AttachUserPolicy', 'CreateRole']),
                'sourceIPAddress': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'userAgent': 'Boto3/1.0.0',
                'errorCode': None,
                'responseElements': {'success': True},
                'requestParameters': {'userName': f'temp-user-{i}', 'policyArn': 'arn:aws:iam::aws:policy/AdministratorAccess'},
                'recipientAccountId': '123456789012',
                'awsRegion': 'us-east-1',
                'eventSource': 'iam.amazonaws.com',
                'attack_scenario': scenario,
                'threat_label': 1,
                'risk_score': np.random.uniform(70, 95)
            }
        
        # Data exfiltration pattern  
        elif scenario == 'data_exfiltration':
            event = {
                'eventTime': (datetime.now() - timedelta(hours=np.random.randint(1, 24))).isoformat(),
                'eventName': np.random.choice(['GetObject', 'ListBuckets', 'PutBucketPolicy']),
                'sourceIPAddress': f"203.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}",  # External IP
                'userAgent': 'python-requests/2.25.1',
                'errorCode': None,
                'responseElements': {'bytesTransferred': np.random.randint(1000000, 10000000)},
                'requestParameters': {'bucketName': 'sensitive-data-bucket'},
                'recipientAccountId': '123456789012',
                'awsRegion': 'us-east-1',
                'eventSource': 's3.amazonaws.com',
                'attack_scenario': scenario,
                'threat_label': 1,
                'risk_score': np.random.uniform(80, 98)
            }
        
        # Reconnaissance pattern
        else:
            event = {
                'eventTime': (datetime.now() - timedelta(hours=np.random.randint(1, 12))).isoformat(),
                'eventName': np.random.choice(['DescribeInstances', 'ListUsers', 'GetUser', 'DescribeSecurityGroups']),
                'sourceIPAddress': f"45.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'userAgent': 'aws-cli/1.18.0',
                'errorCode': 'AccessDenied' if np.random.random() > 0.7 else None,
                'responseElements': None if np.random.random() > 0.7 else {'instancesSet': []},
                'requestParameters': {'maxResults': 1000},
                'recipientAccountId': '123456789012',
                'awsRegion': 'us-east-1',
                'eventSource': f"{np.random.choice(['ec2', 'iam'])}.amazonaws.com",
                'attack_scenario': scenario,
                'threat_label': 1,
                'risk_score': np.random.uniform(60, 85)
            }
        
        attack_events.append(event)
    
    # Combine all events
    all_events = normal_events + attack_events
    np.random.shuffle(all_events)
    
    return pd.DataFrame(all_events)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Threat Detection and Log Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MATERIAL 3 EXPRESSIVE STYLING ---
st.markdown("""
<style>
    /* Import Arial font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    body { 
        font-family: 'Arial', sans-serif; 
        color: #262730; 
        background-color: #FFFFFF;
    }
    
    .main {
        font-family: 'Arial', sans-serif;
        background-color: #FFFFFF;
    }
    
    /* Hide sidebar completely */
    .css-1d391kg, .css-1lcbmhc, .css-1outpf7, section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #003087 0%, #002669 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 8px rgba(0, 48, 135, 0.15);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
        margin: 0;
    }
    
    /* Material-like card styling */
    .material-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        transition: all 0.3s ease;
        border: 1px solid #f0f0f0;
    }
    
    .material-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    .material-metric-value { 
        font-size: 2.5em; 
        font-weight: 700; 
        color: #003087; 
        margin-top: 5px; 
    }
    
    .material-metric-label { 
        font-size: 0.9em; 
        color: #6C757D; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
    }
    
    /* Threat level specific styling */
    .threat-critical {
        border-left: 5px solid #dc3545;
        background: linear-gradient(145deg, #fff5f5, #ffe6e6);
    }
    
    .threat-high {
        border-left: 5px solid #fd7e14;
        background: linear-gradient(145deg, #fff8f0, #ffebcc);
    }
    
    .threat-medium {
        border-left: 5px solid #ffc107;
        background: linear-gradient(145deg, #fffbf0, #fff3cd);
    }
    
    .threat-low {
        border-left: 5px solid #28a745;
        background: linear-gradient(145deg, #f8fff8, #d4edda);
    }
    
    /* Upload section styling */
    .upload-section {
        background-color: #FFFFFF;
        border: 2px dashed #003087;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #002669;
        background-color: #f8f9fa;
    }
    
    /* Button styling with accessibility */
    .stButton > button {
        background-color: #003087;
        color: white;
        border-radius: 50px;
        padding: 12px 24px;
        font-size: 16px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        font-weight: 600;
        font-family: 'Arial', sans-serif;
    }
    
    .stButton > button:hover {
        background-color: #002669;
        transform: scale(1.05);
    }
    
    .stButton > button:active {
        border: 2px solid #FFFFFF;
    }
    
    /* Chart container styling */
    .chart-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        padding: 20px;
        margin-bottom: 25px;
        border: 1px solid #f0f0f0;
    }
    
    /* Section headers */
    .section-header {
        color: #003087;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #003087;
    }
    
    /* Info cards */
    .info-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #003087;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(145deg, #d4edda, #ffffff);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 600px) {
        .material-card {
            padding: 15px;
            border-radius: 8px;
        }
        .material-metric-value { 
            font-size: 2em; 
        }
        .stButton > button { 
            padding: 10px 20px; 
            font-size: 14px; 
        }
        .main-header h1 {
            font-size: 2rem;
        }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = 'aws_threat_detection_model.pkl'
    # Try different possible paths for the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        model_path,
        os.path.join('src', model_path),
        os.path.join(current_dir, model_path),
        os.path.join(current_dir, 'src', model_path)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"Error loading model from {path}: {e}")
                continue
    
    # If no model file found, create a mock model for demonstration
    st.info("⚠️ Model file not found. Using mock model for demonstration purposes.")
    
    # Create mock model structure
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a simple mock model
    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create mock feature columns (these should match what extract_features produces)
    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'is_night',
        'is_private_ip', 'ip_first_octet', 'user_agent_length',
        'is_boto3', 'is_console', 'is_cli', 'has_error',
        'is_access_denied', 'is_create_event', 'is_delete_event',
        'is_list_event', 'is_admin_event',
        'is_external_ip', 'is_sensitive_api', 'suspicious_user_agent',
        'eventName_encoded', 'eventSource_encoded', 'awsRegion_encoded'
    ]
    
    # Create mock training data and fit the model
    X_mock = np.random.rand(100, len(feature_cols))
    y_mock = np.random.randint(0, 2, 100)
    mock_model.fit(X_mock, y_mock)
    
    # Create mock encoders
    mock_encoders = {
        'eventName': LabelEncoder(),
        'eventSource': LabelEncoder(),
        'awsRegion': LabelEncoder()
    }
    
    # Fit encoders with some common AWS values
    mock_encoders['eventName'].fit(['AssumeRole', 'GetSessionToken', 'CreateUser', 'DeleteUser', 'ListUsers', 'RunInstances', 'StopInstances', 'StartInstances', 'TerminateInstances', 'DescribeInstances', 'CreateBucket', 'DeleteBucket', 'PutObject', 'GetObject', 'ListBuckets'])
    mock_encoders['eventSource'].fit(['iam.amazonaws.com', 'sts.amazonaws.com', 's3.amazonaws.com', 'ec2.amazonaws.com', 'cloudtrail.amazonaws.com'])
    mock_encoders['awsRegion'].fit(['us-east-1', 'us-east-2', 'us-west-1', 'us-west-2', 'eu-west-1'])
    
    return {
        'model': mock_model,
        'feature_columns': feature_cols,
        'label_encoders': mock_encoders,
        'model_metrics': {
            'feature_importance': [
                {'feature': col, 'importance': np.random.rand()} 
                for col in feature_cols
            ]
        }
    }

try:
    model_pkg = load_model()
    rf_model = model_pkg['model']
    feature_columns = model_pkg['feature_columns']
    encoders = model_pkg['label_encoders']
    feature_importance_df = pd.DataFrame(model_pkg['model_metrics']['feature_importance'])
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- PREDICTION HELPER ---
def predict_df(df_events):
    """Batch predictions for a DataFrame of events."""
    try:
        df_feat = extract_features(df_events, encoders)
        
        # Select and reorder columns to match feature_columns
        X = df_feat[feature_columns]
        
        # Check for any remaining NaN values before prediction
        if X.isnull().any().any():
            st.warning("NaN values detected in features before prediction. Filling with 0.")
            X = X.fillna(0)

        probs = rf_model.predict_proba(X)[:, 1]
        preds = rf_model.predict(X)
        return preds, probs
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return np.zeros(len(df_events)), np.zeros(len(df_events))

# --- HELPER FUNCTIONS ---
def parse_json_file(uploaded_file):
    """Parse JSON file with better error handling"""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Try to parse as JSON
        data = json.loads(content)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if 'Records' in data:
                # CloudTrail format
                return pd.DataFrame(data['Records'])
            else:
                # Single record
                return pd.DataFrame([data])
        elif isinstance(data, list):
            # List of records
            return pd.DataFrame(data)
        else:
            st.error("Unsupported JSON format")
            return None
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {e}")
        return None
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        return None

def get_risk_level_info(probability):
    """Get risk level information with styling"""
    if probability >= 0.8:
        return "🔴 CRITICAL", "threat-critical"
    elif probability >= 0.6:
        return "🟠 HIGH", "threat-high"
    elif probability >= 0.4:
        return "🟡 MEDIUM", "threat-medium"
    else:
        return "🟢 LOW", "threat-low"

# --- MAIN LAYOUT ---
st.markdown('<div class="main-header"><h1>🛡️ AI Threat Detection and Log Analysis</h1><p>Advanced CloudTrail Security Monitoring Platform</p></div>', unsafe_allow_html=True)

# --- BATCH ANALYSIS SECTION ---
st.markdown('<h2 class="section-header">📊 Batch Log Analysis</h2>', unsafe_allow_html=True)

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.subheader("📁 Upload CloudTrail Log File")
st.write("Supported formats: JSON (CloudTrail format), CSV")

uploaded = st.file_uploader("Choose a file", type=['json', 'csv'], key="batch_upload")

if uploaded:
    try:
        # Show file info
        st.info(f"📄 File: {uploaded.name} ({uploaded.size} bytes)")
        
        # Parse file based on type
        if uploaded.name.endswith('.json'):
            df = parse_json_file(uploaded)
        else:
            df = pd.read_csv(uploaded)
        
        if df is not None and not df.empty:
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            # Show preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Perform predictions
            with st.spinner("🔄 Analyzing threats..."):
                preds, probs = predict_df(df)
                df['prediction'] = preds
                df['threat_probability'] = probs
                df['risk_level'] = [get_risk_level_info(p)[0] for p in probs]
            
            # Show summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_events = len(df)
                st.markdown(f'<div class="material-card"><div class="material-metric-label">Total Events</div><div class="material-metric-value">{total_events}</div></div>', unsafe_allow_html=True)
            
            with col2:
                threats_detected = sum(preds)
                st.markdown(f'<div class="material-card threat-critical"><div class="material-metric-label">Threats Detected</div><div class="material-metric-value">{threats_detected}</div></div>', unsafe_allow_html=True)
            
            with col3:
                avg_risk = np.mean(probs)
                st.markdown(f'<div class="material-card"><div class="material-metric-label">Average Risk</div><div class="material-metric-value">{avg_risk:.1%}</div></div>', unsafe_allow_html=True)
            
            with col4:
                high_risk = sum(probs >= 0.6)
                st.markdown(f'<div class="material-card threat-medium"><div class="material-metric-label">High Risk Events</div><div class="material-metric-value">{high_risk}</div></div>', unsafe_allow_html=True)
            
            # Show detailed results
            st.subheader("📊 Analysis Results")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                show_threats_only = st.checkbox("Show threats only", value=False)
            with col2:
                min_probability = st.slider("Minimum threat probability", 0.0, 1.0, 0.0, 0.1)
            
            # Apply filters
            filtered_df = df.copy()
            if show_threats_only:
                filtered_df = filtered_df[filtered_df['prediction'] == 1]
            filtered_df = filtered_df[filtered_df['threat_probability'] >= min_probability]
            
            # Sort by threat probability
            filtered_df = filtered_df.sort_values('threat_probability', ascending=False)
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download results
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"threat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.error("❌ Failed to load data from the uploaded file")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("💡 Please ensure your file is in the correct format (CloudTrail JSON or CSV)")

st.markdown('</div>', unsafe_allow_html=True)

# --- DATA VISUALIZATION SECTION ---
st.markdown('<h2 class="section-header">📈 Data Visualization & Analytics</h2>', unsafe_allow_html=True)

st.write("Generate and visualize synthetic CloudTrail data for analysis and demonstration")

if st.button("🎲 Generate Sample Data for Visualization", type="primary"):
    with st.spinner("🔄 Generating visualization data..."):
        # Generate sample data
        viz_df = generate_synthetic_cloudtrail_data(1000)
        
        # Create comprehensive visualizations
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Threat Distribution', 'Risk Score Distribution', 
                           'Event Timeline', 'Top Attack APIs'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Threat distribution pie chart
        threat_counts = viz_df['threat_label'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Normal', 'Threat'], values=threat_counts.values,
                   marker_colors=['#28a745', '#dc3545']),
            row=1, col=1
        )
        
        # Risk score histogram
        fig.add_trace(
            go.Histogram(x=viz_df['risk_score'], name='Risk Score Distribution',
                        marker_color='#003087'),
            row=1, col=2
        )
        
        # Timeline of events
        viz_df['eventTime'] = pd.to_datetime(viz_df['eventTime'])
        timeline_data = viz_df.groupby(viz_df['eventTime'].dt.date)['threat_label'].sum().reset_index()
        fig.add_trace(
            go.Scatter(x=timeline_data['eventTime'], y=timeline_data['threat_label'],
                      mode='lines+markers', name='Daily Threats',
                      marker_color='#dc3545'),
            row=2, col=1
        )
        
        # Top attack APIs
        attack_apis = viz_df[viz_df['threat_label'] == 1]['eventName'].value_counts().head(5)
        fig.add_trace(
            go.Bar(x=attack_apis.index, y=attack_apis.values,
                   name='Attack APIs', marker_color='#003087'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="AWS CloudTrail Security Analysis Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # IP Address Analysis
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🌐 IP Address Analysis")
            ip_analysis = viz_df.groupby('threat_label')['sourceIPAddress'].apply(
                lambda x: x.str.startswith(('10.', '192.168.')).sum()
            ).reset_index()
            ip_analysis.columns = ['threat_label', 'private_ip_count']
            ip_analysis['threat_type'] = ip_analysis['threat_label'].map({0: 'Normal', 1: 'Threat'})
            
            fig_ip = px.bar(ip_analysis, x='threat_type', y='private_ip_count', 
                           title="Private IP Usage by Threat Type",
                           color='threat_type', color_discrete_map={'Normal': '#28a745', 'Threat': '#dc3545'})
            st.plotly_chart(fig_ip, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Event Source Analysis
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🔧 Event Source Analysis")
            source_analysis = viz_df.groupby(['eventSource', 'threat_label']).size().reset_index(name='count')
            source_analysis['threat_type'] = source_analysis['threat_label'].map({0: 'Normal', 1: 'Threat'})
            
            fig_source = px.bar(source_analysis, x='eventSource', y='count', 
                               color='threat_type', title="Events by Source and Threat Type",
                               color_discrete_map={'Normal': '#28a745', 'Threat': '#dc3545'})
            fig_source.update_xaxes(tickangle=45)
            st.plotly_chart(fig_source, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show data summary
        st.subheader("📋 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="material-card"><div class="material-metric-label">Total Events</div><div class="material-metric-value">{len(viz_df)}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="material-card"><div class="material-metric-label">Normal Events</div><div class="material-metric-value">{len(viz_df[viz_df["threat_label"] == 0])}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="material-card threat-critical"><div class="material-metric-label">Threat Events</div><div class="material-metric-value">{len(viz_df[viz_df["threat_label"] == 1])}</div></div>', unsafe_allow_html=True)
        with col4:
            threat_rate = len(viz_df[viz_df['threat_label'] == 1]) / len(viz_df) * 100
            st.markdown(f'<div class="material-card"><div class="material-metric-label">Threat Rate</div><div class="material-metric-value">{threat_rate:.1f}%</div></div>', unsafe_allow_html=True)


