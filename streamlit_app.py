#!/usr/bin/env python3
"""
Streamlit dashboard for ArXiv citation analysis.

Run with: streamlit run streamlit_app.py
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ArXiv Citation Analysis", layout="wide")
st.title("📊 ArXiv Citation Prediction Dashboard")
st.markdown("Predict how many citations an arXiv paper will receive based on abstract quality, content, and metadata.")

# ===========================================================================
# Load data and models
# ===========================================================================

@st.cache_resource
def load_assets():
    """Load parquet files and trained models."""
    features = pd.read_parquet("features.parquet")
    targets = pd.read_parquet("targets.parquet")
    arxiv_ids = pd.read_parquet("arxiv_ids.parquet")

    # Load TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vec = pickle.load(f)

    # Load LASSO model and scaler
    with open("lasso_model.pkl", "rb") as f:
        models = pickle.load(f)
    scaler = models['scaler']
    lasso_model = models['lasso']
    lasso_coefs = models['lasso_coefs']
    feature_names = models['feature_names']

    return {
        'features': features,
        'targets': targets,
        'arxiv_ids': arxiv_ids,
        'tfidf_vec': tfidf_vec,
        'scaler': scaler,
        'lasso': lasso_model,
        'lasso_coefs': lasso_coefs,
        'feature_names': feature_names,
    }

try:
    assets = load_assets()
    features = assets['features']
    targets = assets['targets']
    arxiv_ids = assets['arxiv_ids']
    tfidf_vec = assets['tfidf_vec']
    scaler = assets['scaler']
    lasso_model = assets['lasso']
    lasso_coefs = assets['lasso_coefs']
    feature_names = assets['feature_names']
except FileNotFoundError as e:
    st.error(f"❌ Missing required files. Please run the pipeline first:\n\n"
             f"1. `python feature_engineering.py` (or `--demo`)\n"
             f"2. `python model_results.py` (or `--demo`)\n\n"
             f"Error: {e}")
    st.stop()

# ===========================================================================
# Sidebar navigation
# ===========================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a panel:",
    ["🔍 Abstract Analyzer", "📈 Feature Importance", "📊 Citation Distribution", "🎯 Model Performance"]
)

# ===========================================================================
# Panel 1: Abstract Analyzer
# ===========================================================================

if page == "🔍 Abstract Analyzer":
    st.header("Abstract Analyzer")
    st.markdown("Paste an abstract (and optionally a title) to predict citation percentile and see which words drive the prediction.")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_abstract = st.text_area(
            "Paste abstract here:",
            height=200,
            placeholder="We propose a novel approach to..."
        )
        user_title = st.text_input(
            "Paper title (optional):",
            placeholder="Short title of your paper"
        )

    if user_abstract.strip():
        # Preprocess input
        combined_text = (user_title if user_title else "") + " " + user_abstract

        # Extract TF-IDF features
        try:
            tfidf_features = tfidf_vec.transform([combined_text])
            tfidf_names = [f"tfidf_{name}" for name in tfidf_vec.get_feature_names_out()]

            # Create feature matrix (only TF-IDF, fill metadata with zeros/means)
            X_user = np.zeros((1, len(feature_names)))
            for i, fname in enumerate(feature_names):
                if fname in tfidf_names:
                    idx = tfidf_names.index(fname)
                    X_user[0, i] = tfidf_features[0, idx]
                elif fname not in tfidf_names and not fname.startswith("tfidf_"):
                    # Use median/mode for metadata features we can't compute
                    X_user[0, i] = features[fname].median() if fname in features.columns else 0.0

            # Scale and predict
            X_user_scaled = scaler.transform(X_user)
            pred_log = lasso_model.predict(X_user_scaled)[0]
            pred_raw = np.expm1(pred_log)

            # Compute percentile
            all_citations = targets['citations_24mo'].values
            percentile = stats.percentileofscore(all_citations, pred_raw)

            # Compute confidence interval (rough: ±1 standard error)
            residuals = (targets['citations_24mo_log'].values - lasso_model.predict(scaler.transform(features.values)))
            std_error = residuals.std()
            conf_lower = np.expm1(pred_log - 1.96 * std_error)
            conf_upper = np.expm1(pred_log + 1.96 * std_error)

            # Display results
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Citations (24mo)", f"{pred_raw:.0f}")
            with col2:
                st.metric("Citation Percentile", f"{percentile:.1f}%")
            with col3:
                st.metric("Confidence Interval (95%)", f"{conf_lower:.0f}–{conf_upper:.0f}")

            # Find and highlight words with LASSO coefficients
            st.markdown("#### Word-Level Contributions (LASSO Coefficients)")

            text_lower = combined_text.lower()
            word_scores = {}

            for feat in tfidf_names:
                word = feat.replace("tfidf_", "")
                coef = lasso_coefs.get(feat, 0.0)
                if coef != 0.0 and word in text_lower:
                    word_scores[word] = coef

            if word_scores:
                sorted_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

                st.markdown("**Top 10 words in your text by LASSO weight:**")
                cols = st.columns(2)
                pos_words = [(w, c) for w, c in sorted_words if c > 0]
                neg_words = [(w, c) for w, c in sorted_words if c < 0]

                with cols[0]:
                    if pos_words:
                        st.markdown("**🟢 Increases citations:**")
                        for word, coef in pos_words:
                            st.write(f"- `{word}` ({coef:+.4f})")

                with cols[1]:
                    if neg_words:
                        st.markdown("**🔴 Decreases citations:**")
                        for word, coef in neg_words:
                            st.write(f"- `{word}` ({coef:+.4f})")
            else:
                st.info("No high-impact words from the LASSO model found in your text.")

        except Exception as e:
            st.error(f"Error processing abstract: {e}")

    else:
        st.info("👆 Paste an abstract to get started.")

# ===========================================================================
# Panel 2: Feature Importance
# ===========================================================================

elif page == "📈 Feature Importance":
    st.header("Feature Importance — LASSO Predictors")
    st.markdown("Which features drive citation predictions? Here are the top 20 positive and negative predictors from LASSO.")

    # Get top positive and negative coefficients
    top_pos = lasso_coefs[lasso_coefs > 0].sort_values(ascending=True).tail(20)
    top_neg = lasso_coefs[lasso_coefs < 0].sort_values(ascending=True).head(20)

    all_top = pd.concat([top_neg, top_pos])
    all_top.index = all_top.index.str.replace("tfidf_", "").str.replace("has_", "has:").str.replace("cat_", "cat:")

    # Plotly bar chart
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
        title="Top 20 Positive and Negative LASSO Coefficients",
        xaxis_title="LASSO Coefficient",
        yaxis_title="Feature",
        height=600,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Legend:**")
    st.markdown("- 🟢 **Green** = predicts MORE citations")
    st.markdown("- 🔴 **Red** = predicts FEWER citations")

# ===========================================================================
# Panel 3: Citation Distribution
# ===========================================================================

elif page == "📊 Citation Distribution":
    st.header("Citation Distribution by Category")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("How are citations distributed across ArXiv categories?")
    with col2:
        log_scale = st.checkbox("Log scale?", value=False)

    # Get categories from features
    cat_cols = [c for c in feature_names if c.startswith("cat_")]
    categories = [c.replace("cat_", "") for c in cat_cols]

    df_plot = targets.copy()
    df_plot['category'] = None

    # Assign categories by looking at feature matrix
    for idx in df_plot.index:
        for i, cat_col in enumerate(cat_cols):
            col_idx = feature_names.index(cat_col)
            if features.iloc[idx, col_idx] > 0.5:
                df_plot.loc[idx, 'category'] = categories[i]
                break

    fig = px.histogram(
        df_plot,
        x='citations_24mo' if not log_scale else 'citations_24mo_log',
        color='category',
        nbins=30,
        title="Citation Distribution (Raw)" if not log_scale else "Citation Distribution (Log Scale)",
        labels={
            'citations_24mo': 'Citations (24mo)',
            'citations_24mo_log': 'log1p(Citations)',
            'category': 'Category'
        },
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Percentile table
    st.markdown("#### Percentile Breakdown by Category")
    percentile_data = []
    for cat in sorted(df_plot['category'].dropna().unique()):
        cat_data = df_plot[df_plot['category'] == cat]['citations_24mo']
        percentile_data.append({
            'Category': cat,
            'p25': cat_data.quantile(0.25),
            'p50 (Median)': cat_data.quantile(0.50),
            'p75': cat_data.quantile(0.75),
            'p90': cat_data.quantile(0.90),
            'p99': cat_data.quantile(0.99),
        })

    percentile_df = pd.DataFrame(percentile_data)
    st.dataframe(percentile_df.set_index('Category').round(1), use_container_width=True)

# ===========================================================================
# Panel 4: Model Performance
# ===========================================================================

elif page == "🎯 Model Performance":
    st.header("Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted Citations")
        X_scaled = scaler.transform(features.values)
        preds = lasso_model.predict(X_scaled)
        preds_raw = np.expm1(preds)
        actuals_raw = targets['citations_24mo'].values

        fig = px.scatter(
            x=actuals_raw,
            y=preds_raw,
            labels={'x': 'Actual Citations', 'y': 'Predicted Citations'},
            title='Predictions vs Actuals',
            height=400,
        )
        # Add diagonal line
        max_val = max(actuals_raw.max(), preds_raw.max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Perfect Prediction', line=dict(dash='dash', color='gray')))
        fig.update_xaxes(title_text='Actual Citations (24mo)')
        fig.update_yaxes(title_text='Predicted Citations (24mo)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Residuals by Category")
        residuals = actuals_raw - preds_raw

        # Get categories
        cat_cols = [c for c in feature_names if c.startswith("cat_")]
        categories = [c.replace("cat_", "") for c in cat_cols]

        residual_data = []
        for idx in range(len(residuals)):
            for i, cat_col in enumerate(cat_cols):
                col_idx = feature_names.index(cat_col)
                if features.iloc[idx, col_idx] > 0.5:
                    residual_data.append({'Category': categories[i], 'Residual': residuals[idx]})
                    break

        residual_df = pd.DataFrame(residual_data)
        fig = px.box(
            residual_df,
            x='Category',
            y='Residual',
            title='Error Distribution by Category',
            height=400,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.markdown("#### Model Performance Metrics")

    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

    mae = mean_absolute_error(targets['citations_24mo_log'].values, preds)
    r2 = r2_score(targets['citations_24mo_log'].values, preds)
    rmse = np.sqrt(mean_squared_error(targets['citations_24mo_log'].values, preds))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score (log scale)", f"{r2:.4f}")
    with col2:
        st.metric("MAE (log scale)", f"{mae:.4f}")
    with col3:
        st.metric("RMSE (log scale)", f"{rmse:.4f}")
    with col4:
        st.metric("Dataset Size", f"{len(features)} papers")

    st.markdown("**Note:** Metrics are on log scale. Real-world performance varies by category.")

# ===========================================================================
# Footer
# ===========================================================================

st.markdown("---")
st.markdown(
    """
    **About this dashboard:**
    - Data: ArXiv papers (2019-2022) with OpenAlex citations
    - Model: LASSO regression on abstract TF-IDF + metadata
    - Limitations: No author h-index, self-citation inflation, category-specific norms
    - For full analysis, see MODEL_REPORT.md
    """
)
