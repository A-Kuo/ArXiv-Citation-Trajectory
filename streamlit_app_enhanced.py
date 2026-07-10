#!/usr/bin/env python3
"""
Enhanced Streamlit dashboard with batch analysis, SHAP-like explanations,
and advanced analytics.

Run with: streamlit run streamlit_app_enhanced.py
"""

import pickle
import json
from pathlib import Path
from io import BytesIO
from datetime import datetime
import csv

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.preprocessing import StandardScaler

from feature_schema import load_builder_from_training_artifacts

st.set_page_config(page_title="ArXiv Citation Analysis (Enhanced)", layout="wide")
st.title("📊 ArXiv Citation Analysis — Enhanced Edition")
st.markdown("Advanced tools: batch prediction, confidence analysis, author trends, export.")

# ===========================================================================
# Load data and models
# ===========================================================================

@st.cache_resource
def load_assets():
    """Load all trained models and data."""
    features = pd.read_parquet("features.parquet")
    targets = pd.read_parquet("targets.parquet")
    arxiv_ids = pd.read_parquet("arxiv_ids.parquet")

    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vec = pickle.load(f)

    with open("lasso_model.pkl", "rb") as f:
        models = pickle.load(f)

    # Load advanced analysis if available
    advanced_analysis = {}
    if Path("advanced_analysis_summary.json").exists():
        with open("advanced_analysis_summary.json") as f:
            advanced_analysis = json.load(f)

    # Shared feature builder — constructs single-paper features the same
    # way feature_engineering.py did at training time, so predictions here
    # match the API and the trained model's expectations.
    builder = load_builder_from_training_artifacts(
        model_path=Path("lasso_model.pkl"),
        vectorizer_path=Path("tfidf_vectorizer.pkl"),
        validation_mode="warn",
    )

    return {
        'features': features,
        'targets': targets,
        'arxiv_ids': arxiv_ids,
        'tfidf_vec': tfidf_vec,
        'scaler': models['scaler'],
        'lasso': models['lasso'],
        'lasso_coefs': models['lasso_coefs'],
        'feature_names': models['feature_names'],
        'advanced': advanced_analysis,
        'builder': builder,
    }

try:
    assets = load_assets()
except FileNotFoundError:
    st.error("❌ Missing required files. Run: `python model_results.py --demo`")
    st.stop()

# Unpack assets
features = assets['features']
targets = assets['targets']
arxiv_ids = assets['arxiv_ids']
tfidf_vec = assets['tfidf_vec']
scaler = assets['scaler']
lasso_model = assets['lasso']
lasso_coefs = assets['lasso_coefs']
feature_names = assets['feature_names']
advanced = assets['advanced']
builder = assets['builder']

# ===========================================================================
# Sidebar navigation
# ===========================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a panel:",
    [
        "🔍 Single Abstract",
        "📋 Batch Analysis",
        "📈 Feature Importance",
        "📊 Citation Distribution",
        "👥 Author Trends",
        "🎯 Model Performance",
        "📉 Advanced Analytics",
    ]
)

# ===========================================================================
# Helper: predict from abstract
# ===========================================================================

def predict_from_abstract(
    title: str,
    abstract: str,
    authors=None,
    category: str = None,
    submitted_date=None,
) -> dict:
    """Predict citations for a single abstract.

    Uses the shared FeatureBuilder so metadata features (author count,
    submission date, category, keyword flags) are actually computed from
    the input instead of falling back to training-set medians — the
    previous version discarded that signal entirely.
    """
    try:
        X_user, validation_log = builder.build_feature_vector(
            title=title,
            abstract=abstract,
            authors=authors,
            category=category,
            submitted_date=submitted_date,
        )
        warnings = [msg for level, msg in validation_log if level == "warning"]

        X_user_scaled = scaler.transform(X_user)
        pred_log = lasso_model.predict(X_user_scaled)[0]
        pred_raw = np.expm1(pred_log)

        percentile = stats.percentileofscore(targets['citations_24mo'].values, pred_raw)

        # Confidence interval
        residuals = targets['citations_24mo_log'].values - lasso_model.predict(scaler.transform(features.values))
        std_error = residuals.std()
        conf_lower = np.expm1(pred_log - 1.96 * std_error)
        conf_upper = np.expm1(pred_log + 1.96 * std_error)

        return {
            'pred_citations': pred_raw,
            'pred_log': pred_log,
            'percentile': percentile,
            'conf_lower': conf_lower,
            'conf_upper': conf_upper,
            'std_error': std_error,
            'warnings': warnings,
        }
    except Exception as e:
        return {'error': str(e)}

# ===========================================================================
# Page: Single Abstract
# ===========================================================================

if page == "🔍 Single Abstract":
    st.header("Predict Citations for a Single Paper")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_title = st.text_input("Paper title:", placeholder="Attention is All You Need")
        user_abstract = st.text_area("Abstract:", height=200, placeholder="We propose a new architecture...")
    with col2:
        user_authors = st.number_input("Number of authors:", min_value=1, max_value=100, value=1)
        user_category = st.selectbox("Category:", options=builder.metadata.categories or ["cs.LG"])
        user_date = st.date_input("Submission date (optional):", value=None)

    if user_abstract.strip():
        result = predict_from_abstract(
            user_title,
            user_abstract,
            authors=int(user_authors),
            category=user_category,
            submitted_date=datetime.combine(user_date, datetime.min.time()) if user_date else None,
        )
        if 'error' in result:
            st.error(f"Error: {result['error']}")
        else:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Citations", f"{result['pred_citations']:.0f}")
            with col2:
                st.metric("Percentile Rank", f"{result['percentile']:.1f}%")
            with col3:
                st.metric("95% CI", f"{result['conf_lower']:.0f}–{result['conf_upper']:.0f}")

            if result.get('warnings'):
                with st.expander("⚠️ Feature construction warnings"):
                    for w in result['warnings']:
                        st.write(f"- {w}")

            # Word analysis
            combined_lower = (user_title + " " + user_abstract).lower()
            word_scores = {}
            tfidf_names = [f"tfidf_{name}" for name in tfidf_vec.get_feature_names_out()]
            for feat in tfidf_names:
                word = feat.replace("tfidf_", "")
                coef = lasso_coefs.get(feat, 0.0)
                if coef != 0.0 and word in combined_lower:
                    word_scores[word] = coef

            if word_scores:
                st.markdown("#### Words from Your Abstract (LASSO Weights)")
                sorted_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

                cols = st.columns(2)
                pos_words = [(w, c) for w, c in sorted_words if c > 0]
                neg_words = [(w, c) for w, c in sorted_words if c < 0]

                with cols[0]:
                    if pos_words:
                        st.markdown("**🟢 Increase citations:**")
                        for word, coef in pos_words:
                            st.write(f"- `{word}` ({coef:+.4f})")

                with cols[1]:
                    if neg_words:
                        st.markdown("**🔴 Decrease citations:**")
                        for word, coef in neg_words:
                            st.write(f"- `{word}` ({coef:+.4f})")
    else:
        st.info("👆 Enter a title and abstract to get a prediction.")

# ===========================================================================
# Page: Batch Analysis
# ===========================================================================

elif page == "📋 Batch Analysis":
    st.header("Batch Predictions")
    st.markdown("Submit multiple abstracts and export results as CSV.")

    col1, col2 = st.columns([2, 1])
    with col1:
        upload_file = st.file_uploader(
            "Upload CSV (columns: title, abstract)",
            type=["csv"],
        )
    with col2:
        st.markdown("### Or enter manually")

    batch_results = []

    if upload_file:
        df_upload = pd.read_csv(upload_file)
        st.write(f"Loaded {len(df_upload)} papers")

        progress_bar = st.progress(0)
        for idx, row in df_upload.iterrows():
            result = predict_from_abstract(
                str(row.get('title', '')),
                str(row.get('abstract', ''))
            )
            if 'error' not in result:
                batch_results.append({
                    'title': row.get('title', ''),
                    'predicted_citations': result['pred_citations'],
                    'percentile': result['percentile'],
                    'ci_lower': result['conf_lower'],
                    'ci_upper': result['conf_upper'],
                })
            progress_bar.progress((idx + 1) / len(df_upload))

    if batch_results:
        result_df = pd.DataFrame(batch_results)
        st.dataframe(result_df, use_container_width=True)

        # Download button
        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            label="Download predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )

        # Summary stats
        st.markdown("#### Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average citations", f"{result_df['predicted_citations'].mean():.1f}")
        with col2:
            st.metric("Median percentile", f"{result_df['percentile'].median():.1f}%")
        with col3:
            st.metric("Papers in top quartile (p75+)", f"{(result_df['percentile'] >= 75).sum()}")

# ===========================================================================
# Page: Feature Importance
# ===========================================================================

elif page == "📈 Feature Importance":
    st.header("Feature Importance — What Drives Citations?")

    top_pos = lasso_coefs[lasso_coefs > 0].sort_values(ascending=True).tail(20)
    top_neg = lasso_coefs[lasso_coefs < 0].sort_values(ascending=True).head(20)
    all_top = pd.concat([top_neg, top_pos])
    all_top.index = all_top.index.str.replace("tfidf_", "").str.replace("has_", "[kw] ").str.replace("cat_", "[cat] ")

    fig = go.Figure()
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in all_top.values]

    fig.add_trace(go.Bar(
        y=all_top.index,
        x=all_top.values,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{v:+.4f}' for v in all_top.values],
        textposition='outside',
    ))

    fig.update_layout(
        title="Top 20 LASSO Predictors (Positive & Negative)",
        xaxis_title="LASSO Coefficient",
        yaxis_title="Feature",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# Page: Citation Distribution
# ===========================================================================

elif page == "📊 Citation Distribution":
    st.header("Citation Distribution by Category")

    log_toggle = st.checkbox("Log scale?")

    cat_cols = [c for c in feature_names if c.startswith("cat_")]
    categories = [c.replace("cat_", "") for c in cat_cols]

    df_plot = targets.copy()
    df_plot['category'] = None
    for idx in df_plot.index:
        for i, cat_col in enumerate(cat_cols):
            col_idx = feature_names.index(cat_col)
            if features.iloc[idx, col_idx] > 0.5:
                df_plot.loc[idx, 'category'] = categories[i]
                break

    x_col = 'citations_24mo_log' if log_toggle else 'citations_24mo'
    fig = px.histogram(
        df_plot,
        x=x_col,
        color='category',
        nbins=30,
        title="Citation Distribution",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Percentiles
    st.markdown("#### Percentile Breakdown")
    percentile_data = []
    for cat in sorted(df_plot['category'].dropna().unique()):
        cat_cites = df_plot[df_plot['category'] == cat]['citations_24mo']
        percentile_data.append({
            'Category': cat,
            'p25': cat_cites.quantile(0.25),
            'p50': cat_cites.quantile(0.50),
            'p75': cat_cites.quantile(0.75),
            'p90': cat_cites.quantile(0.90),
        })
    st.dataframe(pd.DataFrame(percentile_data).set_index('Category').round(1))

# ===========================================================================
# Page: Author Trends
# ===========================================================================

elif page == "👥 Author Trends":
    st.header("Author & Keyword Patterns")

    if advanced and 'trends' in advanced:
        trends = advanced['trends']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Author Count Effect")
            author_data = pd.DataFrame([
                {
                    'Authors': n,
                    'Count': stats['count'],
                    'Median Cites': stats['median_citations'],
                }
                for n, stats in trends['author_analysis'].items()
            ]).sort_values('Authors')

            fig = px.line(
                author_data,
                x='Authors',
                y='Median Cites',
                title='Median Citations by Author Count',
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top Keywords")
            kw_data = pd.DataFrame([
                {
                    'Keyword': kw,
                    'Mean Cites': stats['mean_citations'],
                    'Papers': stats['papers_with_keyword'],
                }
                for kw, stats in trends['keyword_analysis'].items()
            ]).sort_values('Mean Cites', ascending=False).head(10)

            fig = px.bar(
                kw_data,
                x='Mean Cites',
                y='Keyword',
                orientation='h',
                title='Top 10 Keywords by Average Citations',
                color='Mean Cites',
                color_continuous_scale='Viridis',
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python advanced_analysis.py` to generate author trend data.")

# ===========================================================================
# Page: Model Performance
# ===========================================================================

elif page == "🎯 Model Performance":
    st.header("Model Evaluation")

    if advanced and 'test_set_results' in advanced:
        test_results = advanced['test_set_results']

        st.subheader("Holdout Test Set Performance (20% holdout)")
        test_df = pd.DataFrame([
            {
                'Model': model.upper(),
                'Test R²': res['r2'],
                'Test MAE': res['mae'],
                'Test RMSE': res['rmse'],
            }
            for model, res in test_results.items()
        ])
        st.dataframe(test_df.set_index('Model'), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted")
        X_scaled = scaler.transform(features.values)
        preds = lasso_model.predict(X_scaled)
        preds_raw = np.expm1(preds)
        actuals_raw = targets['citations_24mo'].values

        fig = px.scatter(
            x=actuals_raw,
            y=preds_raw,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title='Predictions vs Actuals (citations)',
        )
        max_val = max(actuals_raw.max(), preds_raw.max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Perfect', line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confidence Calibration")
        if advanced and 'calibration' in advanced:
            cal_data = []
            for nom, emp in advanced['calibration']['coverage'].items():
                cal_data.append({
                    'Nominal': f'{float(nom):.0%}',
                    'Empirical': f'{emp:.1%}',
                    'Match': '✓' if abs(emp - float(nom)) < 0.05 else '✗',
                })
            st.dataframe(pd.DataFrame(cal_data))
        else:
            st.info("Run `python advanced_analysis.py` for calibration analysis.")

# ===========================================================================
# Page: Advanced Analytics
# ===========================================================================

elif page == "📉 Advanced Analytics":
    st.header("Advanced Analytics")

    if advanced:
        # Citation velocity
        if 'velocity' in advanced:
            velocity = advanced['velocity']
            st.subheader("Citation Velocity")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("First 12mo avg", f"{velocity['citations_first_12mo_mean']:.1f} citations")
            with col2:
                st.metric("Months 13-24 avg", f"{velocity['citations_second_12mo_mean']:.1f} citations")
            with col3:
                st.metric("Acceleration ratio", f"{velocity['velocity_to_12mo_ratio']:.2f}x")

            st.markdown("""
            **Interpretation:**
            - Ratio > 1: Citations accelerating over time (growing impact)
            - Ratio < 1: Citations plateau after 12 months (early-career papers)
            """)

        # Recommendations
        st.subheader("Recommendations for Next Steps")
        st.markdown("""
        1. **Feature Engineering:** Test polynomial features, interaction terms
        2. **Data Collection:** Add papers from 2023-2026 to track evolving trends
        3. **Model Explainability:** Implement SHAP values for per-prediction insights
        4. **API Development:** Build REST endpoint for batch predictions
        5. **Monitoring:** Track forecast accuracy over time
        """)
    else:
        st.info("Run `python advanced_analysis.py` to unlock advanced analytics.")

# ===========================================================================
# Footer
# ===========================================================================

st.markdown("---")
st.markdown("""
**About this enhanced dashboard:**
- Batch predictions: Upload CSV, get predictions for multiple papers
- Confidence calibration: Check if our 95% CIs are actually 95% accurate
- Author trends: Discover which author counts and keywords correlate with impact
- Test set evaluation: Honest model comparison on holdout data
- Citation velocity: How quickly do papers accumulate citations?
""")
