import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import re
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class ThreatDetector:
    """Core threat detection logic separated from UI"""
    
    def __init__(self, model_path='aws_threat_detection_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_columns = None
        self.encoders = None
        self.feature_importance_df = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and encoders"""
        # Try different possible paths for the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            self.model_path,
            os.path.join('src', self.model_path),
            os.path.join(current_dir, self.model_path),
            os.path.join(current_dir, 'src', self.model_path)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        model_pkg = pickle.load(f)
                        self.model = model_pkg['model']
                        self.feature_columns = model_pkg['feature_columns']
                        self.encoders = model_pkg['label_encoders']
                        self.feature_importance_df = pd.DataFrame(model_pkg['model_metrics']['feature_importance'])
                        return True
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")
                    continue
        
        # Create mock model if no model file found
        self._create_mock_model()
        return False
    
    def _create_mock_model(self):
        """Create a mock model for demonstration purposes"""
        # Create a simple mock model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create mock feature columns
        self.feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'is_private_ip', 'ip_first_octet', 'user_agent_length',
            'is_boto3', 'is_console', 'is_cli', 'has_error',
            'is_access_denied', 'is_create_event', 'is_delete_event',
            'is_list_event', 'is_admin_event',
            'is_external_ip', 'is_sensitive_api', 'suspicious_user_agent',
            'eventName_encoded', 'eventSource_encoded', 'awsRegion_encoded'
        ]
        
        # Create mock training data and fit the model
        X_mock = np.random.rand(100, len(self.feature_columns))
        y_mock = np.random.randint(0, 2, 100)
        self.model.fit(X_mock, y_mock)
        
        # Create mock encoders
        self.encoders = self._create_mock_encoders()
        
        # Create mock feature importance
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': np.random.rand(len(self.feature_columns))
        })
    
    def _create_mock_encoders(self):
        """Create mock label encoders with common AWS values"""
        encoders = {
            'eventName': LabelEncoder(),
            'eventSource': LabelEncoder(),
            'awsRegion': LabelEncoder()
        }
        
        # Common AWS event names
        event_names = ['AssumeRole', 'GetSessionToken', 'CreateUser', 'DeleteUser', 'ListUsers', 
                      'RunInstances', 'StopInstances', 'StartInstances', 'TerminateInstances', 
                      'DescribeInstances', 'CreateBucket', 'DeleteBucket', 'PutObject', 'GetObject', 
                      'ListBuckets', 'ConsoleLogin', 'AssumeRoleWithWebIdentity', 'AssumeRoleWithSAML']
        
        # Common AWS event sources
        event_sources = ['iam.amazonaws.com', 'sts.amazonaws.com', 's3.amazonaws.com', 
                        'ec2.amazonaws.com', 'cloudtrail.amazonaws.com', 'lambda.amazonaws.com', 
                        'rds.amazonaws.com', 'dynamodb.amazonaws.com', 'sns.amazonaws.com']
        
        # Common AWS regions
        regions = ['us-east-1', 'us-east-2', 'us-west-1', 'us-west-2', 'eu-west-1', 
                  'eu-central-1', 'ap-southeast-1', 'ap-northeast-1']
        
        encoders['eventName'].fit(event_names)
        encoders['eventSource'].fit(event_sources)
        encoders['awsRegion'].fit(regions)
        
        return encoders
    
    def extract_features(self, df):
        """Extract features from CloudTrail log DataFrame"""
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
            df_feat['is_private_ip'] = df_feat['sourceIPAddress'].apply(self._is_private_ip)
            df_feat['is_external_ip'] = (~df_feat['sourceIPAddress'].apply(lambda x: self._is_private_ip(x) == 1)).astype(int)
            df_feat['ip_first_octet'] = df_feat['sourceIPAddress'].apply(self._get_first_octet)
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
            df_feat['suspicious_user_agent'] = df_feat['userAgent'].fillna('').apply(self._is_suspicious_user_agent).astype(int)
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
            df_feat['is_sensitive_api'] = df_feat['eventName'].fillna('').apply(self._is_sensitive_api).astype(int)
            df_feat['eventName_encoded'] = df_feat['eventName'].apply(self._encode_safe, encoder='eventName')
        else:
            df_feat['is_create_event'] = 0
            df_feat['is_delete_event'] = 0
            df_feat['is_list_event'] = 0
            df_feat['is_admin_event'] = 0
            df_feat['is_sensitive_api'] = 0
            df_feat['eventName_encoded'] = 0
        
        # Encoded eventSource and awsRegion
        if 'eventSource' in df_feat.columns:
            df_feat['eventSource_encoded'] = df_feat['eventSource'].apply(self._encode_safe, encoder='eventSource')
        else:
            df_feat['eventSource_encoded'] = 0
        
        if 'awsRegion' in df_feat.columns:
            df_feat['awsRegion_encoded'] = df_feat['awsRegion'].apply(self._encode_safe, encoder='awsRegion')
        else:
            df_feat['awsRegion_encoded'] = 0
        
        # Fill any remaining NaN values
        df_feat = df_feat.fillna(0)
        
        return df_feat
    
    def _is_private_ip(self, ip):
        """Check if IP address is private"""
        if pd.isna(ip) or not isinstance(ip, str):
            return 0
        private_patterns = [
            r'^10\.',
            r'^192\.168\.',
            r'^172\.(1[6-9]|2[0-9]|3[0-1])\.'
        ]
        return int(any(re.match(pattern, ip) for pattern in private_patterns))
    
    def _get_first_octet(self, ip):
        """Extract first octet from IP address"""
        if pd.isna(ip) or not isinstance(ip, str):
            return 0
        try:
            return int(ip.split('.')[0])
        except:
            return 0
    
    def _is_suspicious_user_agent(self, user_agent):
        """Check if user agent contains suspicious patterns"""
        if pd.isna(user_agent):
            return False
        suspicious_patterns = ['nmap', 'nessus', 'metasploit', 'sqlmap']
        return any(p in user_agent.lower() for p in suspicious_patterns)
    
    def _is_sensitive_api(self, event_name):
        """Check if event name is a sensitive API call"""
        if pd.isna(event_name):
            return False
        sensitive_apis = ['DeleteUser', 'DeleteRole', 'AttachUserPolicy', 'AttachRolePolicy', 
                         'PutGroupPolicy', 'UpdateAccessKey']
        return event_name in sensitive_apis
    
    def _encode_safe(self, value, encoder):
        """Safely encode categorical values"""
        if pd.isna(value) or encoder not in self.encoders:
            return -1
        try:
            if value in self.encoders[encoder].classes_:
                return self.encoders[encoder].transform([value])[0]
            else:
                return -1
        except:
            return -1
    
    def predict_single(self, event_data):
        """Predict threat for a single event"""
        df = pd.DataFrame([event_data])
        return self.predict_batch(df)
    
    def predict_batch(self, df_events):
        """Batch predictions for a DataFrame of events"""
        try:
            df_feat = self.extract_features(df_events)
            
            # Select only the features that exist in both the dataframe and the model
            available_features = [col for col in self.feature_columns if col in df_feat.columns]
            missing_features = [col for col in self.feature_columns if col not in df_feat.columns]
            
            # Add missing features with default values
            for feature in missing_features:
                df_feat[feature] = 0
            
            X = df_feat[self.feature_columns]
            probs = self.model.predict_proba(X)[:, 1]
            preds = self.model.predict(X)
            
            return preds, probs, missing_features
        except Exception as e:
            # Return default values on error
            return np.zeros(len(df_events)), np.zeros(len(df_events)), []
    
    def get_risk_level(self, probability):
        """Get risk level information"""
        if probability >= 0.8:
            return "CRITICAL", "🔴"
        elif probability >= 0.6:
            return "HIGH", "🟠"
        elif probability >= 0.4:
            return "MEDIUM", "🟡"
        else:
            return "LOW", "🟢"
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        if self.feature_importance_df is not None:
            return self.feature_importance_df.nlargest(top_n, 'importance')
        return pd.DataFrame()

class DataProcessor:
    """Handle file parsing and data processing"""
    
    @staticmethod
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
                    return pd.DataFrame(data['Records']), None
                else:
                    # Single record
                    return pd.DataFrame([data]), None
            elif isinstance(data, list):
                # List of records
                return pd.DataFrame(data), None
            else:
                return None, "Unsupported JSON format"
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON format: {e}"
        except Exception as e:
            return None, f"Error parsing JSON: {e}"
    
    @staticmethod
    def parse_csv_file(uploaded_file):
        """Parse CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            return df, None
        except Exception as e:
            return None, f"Error parsing CSV: {e}"
    
    @staticmethod
    def validate_data(df):
        """Validate that the DataFrame has required columns"""
        required_columns = ['eventTime', 'eventName', 'sourceIPAddress']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        return True, "Data validation passed"
