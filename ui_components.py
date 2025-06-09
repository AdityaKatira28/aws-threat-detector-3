import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

class UIComponents:
    """UI components and styling for the Streamlit app"""
    
    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="AWS Threat Detection AI Demo",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @staticmethod
    def load_custom_css():
        """Load custom CSS styling"""
        st.markdown("""
        <style>
            .main-header {
                background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                text-align: center;
                color: white;
            }
            .metric-card {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #007bff;
                margin: 0.5rem 0;
            }
            .threat-high {
                border-left-color: #dc3545 !important;
                background: #fff5f5;
            }
            .threat-medium {
                border-left-color: #ffc107 !important;
                background: #fffbf0;
            }
            .threat-low {
                border-left-color: #28a745 !important;
                background: #f8fff8;
            }
            .upload-section {
                background: #f8f9fa;
                padding: 2rem;
                border-radius: 10px;
                margin: 1rem 0;
                border: 2px dashed #007bff;
            }
            .stButton > button {
                background: linear-gradient(90deg, #007bff, #0056b3);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 0.5rem 1rem;
                font-weight: bold;
            }
            .sidebar .sidebar-content {
                background: #f8f9fa;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render the main header"""
        st.markdown(
            '<div class="main-header">'
            '<h1>🛡️ AWS Threat Detection AI Demo</h1>'
            '<p>Advanced CloudTrail Log Analysis for Security Monitoring</p>'
            '</div>', 
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_metric_card(value, label, risk_class=""):
        """Render a metric card"""
        return f'<div class="metric-card {risk_class}"><h3>{value}</h3><p>{label}</p></div>'
    
    @staticmethod
    def render_file_upload_section():
        """Render the file upload section"""
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📁 Upload CloudTrail Log File")
        st.write("Supported formats: JSON (CloudTrail format), CSV")
        
        uploaded = st.file_uploader("Choose a file", type=['json', 'csv'], key="batch_upload")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return uploaded
    
    @staticmethod
    def show_file_info(uploaded_file):
        """Show information about uploaded file"""
        st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    @staticmethod
    def show_data_preview(df, max_rows=10):
        """Show data preview in an expander"""
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(max_rows), use_container_width=True)
    
    @staticmethod
    def render_summary_metrics(total_events, threats_detected, avg_risk, high_risk):
        """Render summary metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                UIComponents.render_metric_card(total_events, "Total Events"), 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                UIComponents.render_metric_card(threats_detected, "Threats Detected", "threat-high"), 
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                UIComponents.render_metric_card(f"{avg_risk:.1%}", "Average Risk"), 
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                UIComponents.render_metric_card(high_risk, "High Risk Events", "threat-medium"), 
                unsafe_allow_html=True
            )
    
    @staticmethod
    def render_filter_controls():
        """Render filter controls for results"""
        col1, col2 = st.columns(2)
        with col1:
            show_threats_only = st.checkbox("Show threats only", value=False)
        with col2:
            min_probability = st.slider("Minimum threat probability", 0.0, 1.0, 0.0, 0.1)
        
        return show_threats_only, min_probability
    
    @staticmethod
    def create_download_button(df, filename_prefix="threat_analysis"):
        """Create download button for results"""
        csv = df.to_csv(index=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )

class SingleEventForm:
    """Handle single event analysis form"""
    
    def __init__(self, encoders):
        self.encoders = encoders
    
    def render_form(self):
        """Render the single event analysis form"""
        st.subheader("🔍 Single Event Analysis")
        st.write("Analyze individual CloudTrail events for threat detection")
        
        with st.form("single_event_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                event_time = st.text_input("🕐 Event Time (ISO)", "2024-06-08T02:30:00Z")
                event_name = st.selectbox("📋 Event Name", list(self.encoders['eventName'].classes_))
                source_ip = st.text_input("🌐 Source IP Address", "203.0.113.45")
                user_agent = st.text_input("🖥️ User Agent", "python-requests/2.25.1")
            
            with col2:
                error_code = st.text_input("⚠️ Error Code (optional)", "")
                event_source = st.selectbox("🔧 Event Source", list(self.encoders['eventSource'].classes_))
                aws_region = st.selectbox("🌍 AWS Region", list(self.encoders['awsRegion'].classes_))
            
            submit = st.form_submit_button("🔍 Analyze Event", use_container_width=True)
        
        if submit:
            event_data = {
                'eventTime': event_time,
                'eventName': event_name,
                'sourceIPAddress': source_ip,
                'userAgent': user_agent,
                'errorCode': error_code or None,
                'eventSource': event_source,
                'awsRegion': aws_region
            }
            return event_data
        
        return None
    
    def render_single_event_results(self, prediction, probability, risk_level, risk_icon, event_data):
        """Render results for single event analysis"""
        st.subheader("📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Get risk class for styling
        risk_class = self._get_risk_class(probability)
        
        with col1:
            prediction_text = "🚨 THREAT" if prediction == 1 else "✅ NORMAL"
            st.markdown(
                UIComponents.render_metric_card(prediction_text, "Prediction", risk_class), 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                UIComponents.render_metric_card(f"{probability:.1%}", "Threat Probability", risk_class), 
                unsafe_allow_html=True
            )
        
        with col3:
            risk_display = f"{risk_icon} {risk_level}"
            st.markdown(
                UIComponents.render_metric_card(risk_display, "Risk Level", risk_class), 
                unsafe_allow_html=True
            )
        
        # Show event details
        with st.expander("📋 Event Details", expanded=True):
            event_details = {
                "Event Time": event_data['eventTime'],
                "Event Name": event_data['eventName'],
                "Source IP": event_data['sourceIPAddress'],
                "User Agent": event_data['userAgent'],
                "Error Code": event_data['errorCode'] or "None",
                "Event Source": event_data['eventSource'],
                "AWS Region": event_data['awsRegion']
            }
            
            for key, value in event_details.items():
                st.write(f"**{key}:** {value}")
    
    def _get_risk_class(self, probability):
        """Get CSS class for risk level styling"""
        if probability >= 0.8:
            return "threat-high"
        elif probability >= 0.6:
            return "threat-high"
        elif probability >= 0.4:
            return "threat-medium"
        else:
            return "threat-low"

class ModelInsights:
    """Handle model insights and visualization"""
    
    @staticmethod
    def render_insights(feature_importance_df):
        """Render model insights tab"""
        st.subheader("📈 Model Insights")
        st.write("Understanding how the AI model makes decisions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ModelInsights.render_feature_importance(feature_importance_df)
        
        with col2:
            ModelInsights.render_model_info()
    
    @staticmethod
    def render_feature_importance(feature_importance_df):
        """Render feature importance chart"""
        st.subheader("🎯 Feature Importance")
        if not feature_importance_df.empty:
            # Sort by importance
            feature_importance_sorted = feature_importance_df.sort_values('importance', ascending=True)
            st.bar_chart(feature_importance_sorted.set_index('feature')['importance'])
            st.caption("Features that most influence the model's threat detection decisions")
    
    @staticmethod
    def render_model_info():
        """Render model information"""
        st.subheader("ℹ️ Model Information")
        st.info("""
        **Model Type:** Random Forest Classifier
        
        **Features Used:**
        - Time-based patterns (hour, day of week)
        - IP address characteristics
        - User agent analysis
        - Event type classification
        - Error patterns
        - AWS service interactions
        
        **Risk Levels:**
        - 🟢 LOW: < 40% threat probability
        - 🟡 MEDIUM: 40-60% threat probability  
        - 🟠 HIGH: 60-80% threat probability
        - 🔴 CRITICAL: > 80% threat probability
        """)

class Sidebar:
    """Handle sidebar content"""
    
    @staticmethod
    def render_sidebar(feature_importance_df):
        """Render the sidebar content"""
        with st.sidebar:
            st.header("🛡️ Threat Detection")
            st.write("Real-time AWS CloudTrail analysis powered by machine learning")
            
            Sidebar.render_quick_stats(feature_importance_df)
            Sidebar.render_tips()
            Sidebar.render_resources()
    
    @staticmethod
    def render_quick_stats(feature_importance_df):
        """Render quick stats section"""
        st.subheader("📊 Quick Stats")
        if not feature_importance_df.empty:
            top_features = feature_importance_df.nlargest(5, 'importance')
            st.write("**Top 5 Important Features:**")
            for _, row in top_features.iterrows():
                st.write(f"• {row['feature']}: {row['importance']:.3f}")
    
    @staticmethod
    def render_tips():
        """Render tips section"""
        st.subheader("💡 Tips")
        st.info("""
        **For Best Results:**
        - Use CloudTrail JSON format
        - Include all standard fields
        - Check for data quality issues
        - Review high-risk events manually
        """)
    
    @staticmethod
    def render_resources():
        """Render resources section"""
        st.subheader("🔗 Resources")
        st.write("""
        - [AWS CloudTrail Documentation](https://docs.aws.amazon.com/cloudtrail/)
        - [Security Best Practices](https://aws.amazon.com/security/)
        - [Threat Detection Guide](https://aws.amazon.com/security/threat-detection/)
        """)
