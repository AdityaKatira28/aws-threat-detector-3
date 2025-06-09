# main_app.py

import streamlit as st
import pandas as pd

from threat_detector_core import ThreatDetector, DataProcessor
from ui_components       import (
    UIComponents,
    SingleEventForm,
    ModelInsights,
    Sidebar
)

def main():
    """Main application function"""

    # ──── 1. PAGE SETUP & STYLING ────
    UIComponents.setup_page_config(
        title="AWS Threat Detection",
        icon="🛡️",
        layout="wide"
    )
    UIComponents.load_custom_css()
    UIComponents.render_header(
        title="AWS Threat Detection Demo",
        subtitle="Analyze your CloudTrail logs for potential security anomalies"
    )

    # ──── 2. INITIALIZE CORE ────
    detector = ThreatDetector(model_path="aws_threat_detection_model.pkl")

    # ──── 3. SIDEBAR ────
    # Show top-10 feature importances in sidebar
    fi_df = detector.get_feature_importance_df(top_n=10)
    Sidebar.render_sidebar(fi_df)

    # ──── 4. MAIN TABS ────
    tab_batch, tab_single, tab_insights = st.tabs(
        ["📁 Batch Analysis", "📄 Single Event", "📊 Model Insights"]
    )

    # ──── Tab 1: Batch Analysis ────
    with tab_batch:
        st.header("📁 Batch Analysis")
        uploaded = st.file_uploader(
            "Upload a CloudTrail file (JSON or CSV)",
            type=["json", "csv"]
        )
        if uploaded:
            # 1️⃣ parse
            if uploaded.name.lower().endswith(".json"):
                df, err = DataProcessor.parse_json_file(uploaded)
            else:
                df, err = DataProcessor.parse_csv_file(uploaded)

            if err:
                st.error(err)
            else:
                # 2️⃣ validate
                valid, msg = DataProcessor.validate_data(df)
                if not valid:
                    st.error(msg)
                else:
                    # 3️⃣ preview & info
                    UIComponents.show_file_info(uploaded)
                    UIComponents.show_data_preview(df)

                    # 4️⃣ predict
                    preds, probs = detector.predict_batch(df)
                    df["prediction"]         = preds
                    df["threat_probability"] = probs

                    # 5️⃣ summary metrics
                    UIComponents.render_summary_metrics(
                        total_events      = len(df),
                        threats_detected  = int((preds == 1).sum()),
                        avg_risk          = probs.mean(),
                        high_risk_count   = int((probs >= 0.6).sum())
                    )

                    # 6️⃣ filtering
                    show_only, min_prob = UIComponents.render_filter_controls(
                        label="Show only events ≥ risk probability:"
                    )
                    out = df
                    if show_only:
                        out = df[df["threat_probability"] >= min_prob]
                    st.dataframe(out)

                    # 7️⃣ download
                    UIComponents.create_download_button(
                        df=out,
                        filename="threats_report.csv",
                        label="Download filtered results"
                    )

    # ──── Tab 2: Single Event ────
    with tab_single:
        st.header("📄 Single Event Analysis")
        form = SingleEventForm(encoders=detector.encoders)
        event_data = form.render_form()
        if event_data is not None:
            preds, probs = detector.predict_single(event_data)
            label, icon = detector.get_risk_level(probs[0])
            form.render_single_event_results(
                prediction   = preds[0],
                probability  = probs[0],
                risk_label   = label,
                risk_icon    = icon,
                raw_event    = event_data
            )

    # ──── Tab 3: Model Insights ────
    with tab_insights:
        st.header("📊 Model Insights")
        fi_df_full = detector.get_feature_importance_df()
        ModelInsights.render_insights(
            fi_df_full,
            title="Feature Importances",
            note="Higher bars → more impactful features"
        )


if __name__ == "__main__":
    main()
