"""
=============================================================================
✈️ FLIGHT PRICE PREDICTION - UNIFIED DASHBOARD
=============================================================================
Complete ML-Powered Analytics & Prediction Platform
Built with Streamlit, XGBoost, and Python
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import pickle
from pathlib import Path
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION (DEPLOYMENT VERSION)
# ============================================================================

# Get current directory (dashboard folder)
current_dir = Path(__file__).parent

# Define paths relative to dashboard folder
MODELS_PATH = current_dir / "models"
DATA_PATH = current_dir / "data" / "processed" / "flight_data_final.csv"

# Verify paths exist
if not MODELS_PATH.exists():
    st.error(f"❌ Models folder not found at: {MODELS_PATH}")
if not DATA_PATH.exists():
    st.error(f"❌ Data file not found at: {DATA_PATH}")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/flight-price-prediction',
        'Report a bug': 'https://github.com/yourusername/flight-price-prediction/issues',
        'About': '# Flight Price Prediction System\nBuilt with ❤️ using Streamlit & XGBoost'
    }
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_custom_css():
    """Load premium custom CSS with modern design"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* ========== GLOBAL STYLES ========== */
    * {
        font-family: 'Inter', 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background with animated gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Content container with glassmorphism */
    .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 25px;
        box-shadow: 0 25px 70px rgba(0,0,0,0.25);
        backdrop-filter: blur(20px);
        margin: 2rem auto;
        max-width: 1500px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* ========== HEADER STYLING ========== */
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
        animation: slideDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin-top: 0.8rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* ========== METRIC CARDS (ENHANCED WITH GLASSMORPHISM) ========== */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeInUp 0.6s ease-out;
        margin: 1rem 0;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-15px) scale(1.05);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.6);
    }
    
    .metric-card h4 {
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card p {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-card small {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* ========== TABS STYLING (MODERN) ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.15);
        color: white;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255,255,255,0.25);
        transform: translateY(-3px);
        border-color: rgba(255,255,255,0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        transform: translateY(-2px);
    }
    
    /* ========== BUTTONS (ENHANCED) ========== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 14px 35px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.7);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-2px);
    }
    
    /* ========== INPUT FIELDS (MODERN) ========== */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 12px 18px;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus,
    .stDateInput > div > div > input:focus,
    .stTimeInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
        outline: none;
    }
    
    /* ========== INFO BOXES (ENHANCED) ========== */
    .info-box {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.1) 0%, rgba(31, 119, 180, 0.05) 100%);
        border-left: 5px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%);
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.1);
    }
    
    /* ========== PREDICTION RESULT (STUNNING) ========== */
    .prediction-result {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        color: white;
        margin: 2.5rem 0;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
        animation: pulse 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-result::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 15s linear infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-60px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ========== RESPONSIVE DESIGN ========== */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2.2rem; }
        .block-container { padding: 1rem; margin: 1rem; }
        .metric-card p { font-size: 2rem; }
        .prediction-result { font-size: 2.5rem; padding: 2rem 1rem; }
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_flight_data():
    """Load flight dataset"""
    try:
        if not DATA_PATH.exists():
            st.warning(f"⚠️ Data file not found at: {DATA_PATH}")
            return None
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load model, metadata, and feature names"""
    try:
        model = joblib.load(MODELS_PATH / "XGBoost_final.pkl")
        
        with open(MODELS_PATH / "model_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        with open(MODELS_PATH / "feature_names.txt", 'r', encoding='utf-8') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        return model, metadata, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def load_performance_data():
    """Load model performance data"""
    try:
        predictions_df = pd.read_csv(MODELS_PATH / "test_predictions.csv")
        comparison_df = pd.read_csv(MODELS_PATH / "model_comparison_results.csv")
        
        return predictions_df, comparison_df
    except Exception as e:
        st.error(f"❌ Error loading performance data: {str(e)}")
        return None, None

@st.cache_data
def load_feature_importance():
    """Load feature importance data"""
    try:
        feature_importance_df = pd.read_csv(MODELS_PATH / "feature_importance.csv")
        return feature_importance_df
    except Exception as e:
        st.warning(f"⚠️ Feature importance data not available: {str(e)}")
        return None
    
# ============================================================================
# HELPER FUNCTIONS FOR GENERIC DATA HANDLING
# ============================================================================

def detect_column_types(df):
    """Intelligently detect column types for generic data handling"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to detect datetime columns from object types
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col].head(100))
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except:
            pass
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }

def get_price_column(df):
    """Detect the price/target column intelligently"""
    price_keywords = ['price', 'cost', 'fare', 'amount', 'value', 'target']
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in price_keywords):
            if df[col].dtype in [np.int64, np.float64]:
                return col
    
    # If not found, return the last numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[-1] if len(numeric_cols) > 0 else None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Flight Price Prediction System</h1>
        <p>Advanced ML-Powered Analytics & Prediction Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_flight_data()
    model, metadata, feature_names = load_model_artifacts()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        st.markdown("---")
        
        # Model status
        if model is not None:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Loaded")
        
        if df is not None:
            st.success(f"✅ Data Loaded ({len(df):,} rows)")
        else:
            st.warning("⚠️ Data Not Loaded")
        
        st.markdown("---")
        
        # Quick stats
        if metadata:
            st.markdown("### 📊 Model Stats")
            st.metric("Accuracy", "94.14%")
            st.metric("MAE", "Rs.606")
            st.metric("Features", metadata['n_features'])
        
        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home",
        "🎯 Predict Price",
        "📊 Model Performance",
        "📈 Feature Analysis",
        "ℹ️ About"
    ])
    
    # TAB 1: HOME
    with tab1:
        show_home_page(df, metadata)
    
    # TAB 2: PREDICTION
    with tab2:
        show_prediction_page(model, metadata, feature_names)
    
    # TAB 3: PERFORMANCE
    with tab3:
        show_performance_page(metadata)
    
    # TAB 4: FEATURE ANALYSIS
    with tab4:
        show_feature_analysis_page(df)
    
    # TAB 5: ABOUT
    with tab5:
        show_about_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>Built with ❤️ using Streamlit, XGBoost, and Python</p>
        <p>© 2026 Flight Price Prediction System | Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 1: HOME
# ============================================================================

def show_home_page(df, metadata):
    """Display enhanced home page with comprehensive EDA"""
    
    st.markdown("## 🏠 Welcome to Flight Price Predictor")
    
    # Hero section
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0;">🎯 About This System</h3>
        <p style="font-size: 1.05rem; line-height: 1.7;">
            This intelligent system uses <strong>advanced machine learning (XGBoost)</strong> to predict flight prices 
            with <strong>94.14% accuracy</strong>. Get instant price estimates for your next journey with 
            confidence intervals and detailed insights!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick metrics (4 columns)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Accuracy</h4>
            <p>94.14%</p>
            <small>R² Score</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Avg Error</h4>
            <p>Rs.606</p>
            <small>7.03% MAPE</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>✅ Reliability</h4>
            <p>92%</p>
            <small>Within ±20%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>⚡ Speed</h4>
            <p>&lt;1s</p>
            <small>Real-time</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== COMPREHENSIVE EDA SECTION ==========
    if df is not None:
        st.markdown("## 📊 Exploratory Data Analysis")
        
        # Detect column types
        col_types = detect_column_types(df)
        price_col = get_price_column(df)
        
        # Dataset overview
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Total Records", f"{len(df):,}")
        with col2:
            st.metric("🔢 Total Features", len(df.columns))
        with col3:
            st.metric("📊 Numeric Columns", len(col_types['numeric']))
        with col4:
            st.metric("🏷️ Categorical Columns", len(col_types['categorical']))
        
        # Sample data display
        st.markdown("### 🔍 Data Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea;">
                <h4 style="margin-top: 0; color: #667eea;">📊 Data Shape</h4>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;"><strong>Rows:</strong> {}</p>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;"><strong>Columns:</strong> {}</p>
            </div>
            """.format(len(df), len(df.columns)), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #28a745;">
                <h4 style="margin-top: 0; color: #28a745;">✅ Data Quality</h4>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;"><strong>Complete:</strong> {:.1f}%</p>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;"><strong>Missing:</strong> {}</p>
            </div>
            """.format((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 
                    df.isnull().sum().sum()), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ffc107;">
                <h4 style="margin-top: 0; color: #ffc107;">📈 Memory Usage</h4>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;"><strong>Size:</strong> {:.2f} MB</p>
            </div>
            """.format(df.memory_usage(deep=True).sum() / 1024**2), unsafe_allow_html=True)

        # Optional: Add expandable detailed view
        with st.expander("🔍 View detailed data sample", expanded=False):
            st.dataframe(df.head(10), use_container_width=True, height=300)

        # Statistics
        if price_col:
            st.markdown(f"### 💰 Price Statistics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📉 Minimum", f"Rs.{df[price_col].min():,.0f}")
            with col2:
                st.metric("📊 Mean", f"Rs.{df[price_col].mean():,.0f}")
            with col3:
                st.metric("📈 Median", f"Rs.{df[price_col].median():,.0f}")
            with col4:
                st.metric("📈 Maximum", f"Rs.{df[price_col].max():,.0f}")
            with col5:
                st.metric("📏 Std Dev", f"Rs.{df[price_col].std():,.0f}")
        
        st.markdown("---")
        
        # ========== VISUALIZATIONS ==========
        st.markdown("### 📈 Data Visualizations")
        
        # Price distribution
        if price_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Price Distribution")
                fig = px.histogram(
                    df, 
                    x=price_col,
                    nbins=50,
                    color_discrete_sequence=['#667eea'],
                    title=f'{price_col} Distribution'
                )
                fig.update_layout(
                    template='plotly_white',
                    height=400,
                    showlegend=False,
                    xaxis_title=price_col,
                    yaxis_title='Frequency'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📦 Price Box Plot")
                fig = px.box(
                    df,
                    y=price_col,
                    color_discrete_sequence=['#764ba2'],
                    title=f'{price_col} Box Plot (Outlier Detection)'
                )
                fig.update_layout(
                    template='plotly_white',
                    height=400,
                    showlegend=False,
                    yaxis_title=price_col
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Categorical analysis
        if len(col_types['categorical']) > 0:
            st.markdown("#### 🏷️ Categorical Features Analysis")
            
            # Select categorical column
            cat_col = st.selectbox(
                "Select categorical feature to analyze:",
                col_types['categorical']
            )
            
            if cat_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Value counts
                    value_counts = df[cat_col].value_counts().head(10)
                    
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        color=value_counts.values,
                        color_continuous_scale='Viridis',
                        title=f'Top 10 {cat_col} Distribution',
                        labels={'x': cat_col, 'y': 'Count'}
                    )
                    fig.update_layout(
                        template='plotly_white',
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Pie chart
                    value_counts_pie = df[cat_col].value_counts().head(8)
                    
                    fig = px.pie(
                        values=value_counts_pie.values,
                        names=value_counts_pie.index,
                        title=f'{cat_col} Proportion',
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig.update_layout(
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Price by categorical feature
        if price_col and len(col_types['categorical']) > 0:
            st.markdown("#### 💰 Price Analysis by Category")
            
            cat_col_price = st.selectbox(
                "Select categorical feature for price analysis:",
                col_types['categorical'],
                key='cat_price'
            )
            
            if cat_col_price:
                # Calculate average price by category
                avg_price = df.groupby(cat_col_price)[price_col].agg(['mean', 'median', 'count']).reset_index()
                avg_price = avg_price.sort_values('mean', ascending=False).head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        avg_price,
                        x=cat_col_price,
                        y='mean',
                        color='mean',
                        color_continuous_scale='Viridis',
                        title=f'Average {price_col} by {cat_col_price}',
                        labels={'mean': f'Average {price_col}'}
                    )
                    fig.update_layout(
                        template='plotly_white',
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(
                        df,
                        x=cat_col_price,
                        y=price_col,
                        color=cat_col_price,
                        title=f'{price_col} Distribution by {cat_col_price}'
                    )
                    fig.update_layout(
                        template='plotly_white',
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        if len(col_types['numeric']) > 1:
            st.markdown("#### 🔥 Correlation Heatmap")
            
            # Calculate correlation
            corr_matrix = df[col_types['numeric']].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title='Feature Correlation Matrix',
                aspect='auto'
            )
            fig.update_layout(
                template='plotly_white',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Data quality report
        st.markdown("### 🔍 Data Quality Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Missing Values")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True, height=300)
            else:
                st.success("✅ No missing values detected!")
        
        with col2:
            st.markdown("#### Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, use_container_width=True, height=300)
    
    else:
        # If no data, show basic info
        st.markdown("### 🌟 Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🎯 Instant Predictions</h4>
                <p style="font-size: 1rem;">Get real-time flight price estimates based on your travel details</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Performance Dashboard</h4>
                <p style="font-size: 1rem;">Explore detailed model performance metrics and visualizations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>📈 Feature Analysis</h4>
                <p style="font-size: 1rem;">Understand which factors influence flight prices the most</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How to use
    st.markdown("### 📖 How to Use This System")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 2.5rem;">1️⃣</h3>
            <h4 style="margin: 0.5rem 0;">Navigate</h4>
            <p style="margin: 0; font-size: 0.9rem;">Go to Predict Price tab</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 2.5rem;">2️⃣</h3>
            <h4 style="margin: 0.5rem 0;">Enter Details</h4>
            <p style="margin: 0; font-size: 0.9rem;">Fill in flight information</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 2.5rem;">3️⃣</h3>
            <h4 style="margin: 0.5rem 0;">Predict</h4>
            <p style="margin: 0; font-size: 0.9rem;">Click predict button</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 2.5rem;">4️⃣</h3>
            <h4 style="margin: 0.5rem 0;">Get Results</h4>
            <p style="margin: 0; font-size: 0.9rem;">View instant predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System information
    if metadata:
        st.markdown("### 🔧 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea;">
                <h4 style="margin-top: 0; color: #667eea;">📊 Model Details</h4>
                <ul style="line-height: 2;">
                    <li><strong>Algorithm:</strong> {metadata['model_type']}</li>
                    <li><strong>Training Samples:</strong> {metadata['training_samples']:,}</li>
                    <li><strong>Test Samples:</strong> {metadata['test_samples']:,}</li>
                    <li><strong>Features:</strong> {metadata['n_features']}</li>
                    <li><strong>Training Date:</strong> {metadata['training_date']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #764ba2;">
                <h4 style="margin-top: 0; color: #764ba2;">🎯 Performance Metrics</h4>
                <ul style="line-height: 2;">
                    <li><strong>Test MAE:</strong> Rs.{metadata['performance_metrics']['test_mae']:,.2f}</li>
                    <li><strong>Test RMSE:</strong> Rs.{metadata['performance_metrics']['test_rmse']:,.2f}</li>
                    <li><strong>Test R²:</strong> {metadata['performance_metrics']['test_r2']:.4f}</li>
                    <li><strong>Test MAPE:</strong> {metadata['performance_metrics']['test_mape']:.2f}%</li>
                    <li><strong>CV MAE:</strong> Rs.{metadata['performance_metrics']['cv_mae']:,.2f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: PREDICTION
# ============================================================================

def show_prediction_page(model, metadata, feature_names):
    """Display prediction interface"""
    
    st.markdown("## 🎯 Flight Price Prediction")
    
    if model is None:
        st.error("❌ Model not loaded. Please check if model files exist in the models directory.")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
        <p>Enter your flight details below to get an instant price prediction powered by our XGBoost model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create form for inputs
    with st.form("prediction_form"):
        st.markdown("### ✈️ Flight Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🏢 Basic Information")
            airline = st.selectbox(
                "Airline",
                options=[
                    'IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 
                    'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
                    'Vistara Premium economy', 'Jet Airways Business',
                    'Multiple carriers Premium economy', 'Trujet'
                ]
            )
            
            source = st.selectbox(
                "From (Source)",
                options=['Delhi', 'Kolkata', 'Banglore', 'Mumbai', 'Chennai']
            )
            
            destination = st.selectbox(
                "To (Destination)",
                options=['Cochin', 'Banglore', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata']
            )
        
        with col2:
            st.markdown("#### 📅 Date & Time")
            journey_date = st.date_input(
                "Journey Date",
                min_value=date.today(),
                value=date.today()
            )
            
            dep_time = st.time_input(
                "Departure Time",
                value=datetime.strptime("09:00", "%H:%M").time()
            )
            
            total_stops = st.selectbox(
                "Number of Stops",
                options=['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']
            )
        
        with col3:
            st.markdown("#### ⏱️ Duration")
            duration_hours = st.number_input(
                "Hours",
                min_value=0,
                max_value=48,
                value=2
            )
            
            duration_minutes = st.number_input(
                "Minutes",
                min_value=0,
                max_value=59,
                value=30,
                step=5
            )
            
            additional_info = st.selectbox(
                "Additional Information",
                options=[
                    'No info', 'In-flight meal not included', 
                    'No check-in baggage included', '1 Long layover',
                    'Change airports', 'Business class'
                ]
            )
        
        st.markdown("---")
        submit_button = st.form_submit_button("🚀 Predict Price", use_container_width=True)
    
    if submit_button:
        with st.spinner("🔮 Analyzing flight details and predicting price..."):
            # Feature engineering (same as before)
            prediction = make_prediction(
                model, feature_names, airline, source, destination,
                journey_date, dep_time, total_stops, duration_hours,
                duration_minutes, additional_info
            )
            
            if prediction is not None:
                display_prediction_results(
                    prediction, airline, source, destination,
                    journey_date, dep_time, total_stops,
                    duration_hours, duration_minutes, additional_info
                )

def make_prediction(model, feature_names, airline, source, destination,
                   journey_date, dep_time, total_stops, duration_hours,
                   duration_minutes, additional_info):
    """Make price prediction with corrected underscore format"""
    try:
        # Feature engineering
        total_duration_minutes = duration_hours * 60 + duration_minutes
        journey_day = journey_date.day
        journey_month = journey_date.month
        journey_dayofweek = journey_date.weekday()
        is_weekend = 1 if journey_dayofweek >= 5 else 0
        dep_hour = dep_time.hour
        dep_minute = dep_time.minute
        
        # Categorize features (using underscores now)
        if 0 <= dep_hour < 6:
            dep_time_category = 'Early_Morning_00_00_06_00'
        elif 6 <= dep_hour < 12:
            dep_time_category = 'Morning_06_00_12_00'
        elif 12 <= dep_hour < 18:
            dep_time_category = 'Afternoon_12_00_18_00'  # May not exist
        else:
            dep_time_category = 'Evening_Night_18_00_24_00'
        
        duration_hours_total = total_duration_minutes / 60
        if duration_hours_total < 2:
            duration_category = 'Very_Short_less_than2h'
        elif duration_hours_total < 5:
            duration_category = 'Short_2_5h'
        elif duration_hours_total < 10:
            duration_category = 'Medium_5_10h'
        elif duration_hours_total < 15:
            duration_category = 'Long_10_15h'  # May not exist
        else:
            duration_category = 'Very_Long_greater_than15h'
        
        stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
        total_stops_encoded = stops_mapping[total_stops]
        
        # Additional features
        route_stops_count = total_stops_encoded
        is_long_haul = 1 if duration_hours_total > 8 else 0
        is_peak_time = 1 if (7 <= dep_hour <= 9) or (17 <= dep_hour <= 19) else 0
        is_holiday_season = 1 if journey_month in [5, 6] else 0
        
        budget_airlines = ['SpiceJet', 'IndiGo', 'Air Asia', 'GoAir']
        is_budget_airline = 1 if airline in budget_airlines else 0
        
        premium_airlines = ['Jet Airways Business', 'Vistara Premium economy', 
                           'Multiple carriers Premium economy']
        is_premium_airline = 1 if airline in premium_airlines else 0
        
        # Additional info (using underscores)
        if 'meal not included' in additional_info.lower():
            additional_info_simplified = 'No_Meal'
        elif 'baggage not included' in additional_info.lower():
            additional_info_simplified = 'No_Baggage'
        elif 'layover' in additional_info.lower():
            additional_info_simplified = 'Layover'
        elif 'business' in additional_info.lower():
            additional_info_simplified = 'Business_Class'
        else:
            additional_info_simplified = 'Standard'
        
        duration_stops_interaction = total_duration_minutes * route_stops_count
        weekend_peak_interaction = is_weekend * is_peak_time
        duration_per_stop = total_duration_minutes / (route_stops_count + 1)
        
        is_outlier = 0
        if (is_premium_airline == 1) or (duration_hours_total > 20) or (total_stops_encoded >= 3):
            is_outlier = 1
        
        # Create feature dictionary
        features = {
            'Journey_Day': journey_day,
            'Journey_Month': journey_month,
            'Journey_DayOfWeek': journey_dayofweek,
            'Is_Weekend': is_weekend,
            'Dep_Hour': dep_hour,
            'Dep_Minute': dep_minute,
            'Duration_Minutes': total_duration_minutes,
            'Duration_Hours': duration_hours_total,
            'Is_Outlier': is_outlier,
            'Route_Stops_Count': route_stops_count,
            'Route_Stops_Match': 1,
            'Is_Peak_Time': is_peak_time,
            'Is_Holiday_Season': is_holiday_season,
            'Is_Long_Haul': is_long_haul,
            'Duration_Per_Stop': duration_per_stop,
            'Is_Budget_Airline': is_budget_airline,
            'Is_Premium_Airline': is_premium_airline,
            'Duration_Stops_Interaction': duration_stops_interaction,
            'Weekend_Peak_Interaction': weekend_peak_interaction,
            'Total_Stops_Encoded': total_stops_encoded
        }
        
        # ✅ Airline mapping (convert display names to feature names with underscores)
        airline_mapping = {
            'Air India': 'Air_India',
            'GoAir': 'GoAir',
            'IndiGo': 'IndiGo',
            'Jet Airways': 'Jet_Airways',
            'Jet Airways Business': 'Jet_Airways_Business',
            'Multiple carriers': 'Multiple_carriers',
            'Multiple carriers Premium economy': 'Multiple_carriers_Premium_economy',
            'SpiceJet': 'SpiceJet',
            'Trujet': 'Trujet',
            'Vistara': 'Vistara',
            'Vistara Premium economy': 'Vistara_Premium_economy'
        }
        
        for airline_display, airline_feature in airline_mapping.items():
            features[f'Airline_{airline_feature}'] = 1 if airline == airline_display else 0
        
        # ✅ Source (no changes needed)
        for source_name in ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']:
            features[f'Source_{source_name}'] = 1 if source == source_name else 0
        
        # ✅ Destination mapping (convert display names to feature names with underscores)
        destination_mapping = {
            'Cochin': 'Cochin',
            'Delhi': 'Delhi',
            'Hyderabad': 'Hyderabad',
            'Kolkata': 'Kolkata',
            'New Delhi': 'New_Delhi'
        }
        
        for dest_display, dest_feature in destination_mapping.items():
            features[f'Destination_{dest_feature}'] = 1 if destination == dest_display else 0
        
        # ✅ Time categories (using underscores - only 3 exist)
        for time_cat in ['Early_Morning_00_00_06_00', 'Evening_Night_18_00_24_00', 'Morning_06_00_12_00']:
            features[f'Dep_Time_Category_{time_cat}'] = 1 if dep_time_category == time_cat else 0
        
        # ✅ Duration categories (using underscores - only 4 exist)
        for dur_cat in ['Medium_5_10h', 'Short_2_5h', 'Very_Long_greater_than15h', 'Very_Short_less_than2h']:
            features[f'Duration_Category_{dur_cat}'] = 1 if duration_category == dur_cat else 0
        
        # ✅ Additional info (using underscores - only 3 exist)
        for info_cat in ['Layover', 'No_Meal', 'Standard']:
            features[f'Additional_Info_Simplified_{info_cat}'] = 1 if additional_info_simplified == info_cat else 0
        
        # ✅ Hour categories (using underscores - 5 exist)
        if 0 <= dep_hour < 4:
            dep_hour_category = 'Late_Night'
        elif 4 <= dep_hour < 8:
            dep_hour_category = 'Early_Morning'
        elif 8 <= dep_hour < 12:
            dep_hour_category = 'Morning'
        elif 12 <= dep_hour < 16:
            dep_hour_category = 'Afternoon'
        elif 16 <= dep_hour < 20:
            dep_hour_category = 'Evening'
        else:
            dep_hour_category = 'Night'
        
        for hour_cat in ['Afternoon', 'Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night']:
            features[f'Dep_Hour_Category_{hour_cat}'] = 1 if dep_hour_category == hour_cat else 0
        
        # Create dataframe
        input_df = pd.DataFrame([features])
        
        # Ensure all features present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder to match training
        input_df = input_df[feature_names]
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        return prediction
        
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def display_prediction_results(prediction, airline, source, destination,
                               journey_date, dep_time, total_stops,
                               duration_hours, duration_minutes, additional_info):
    """Display prediction results"""
    
    std_error = 605.73
    lower_bound = max(0, prediction - 1.96 * std_error)
    upper_bound = prediction + 1.96 * std_error
    
    st.markdown("---")
    
    # Main prediction
    st.markdown(f"""
    <div class="prediction-result">
        <div>Predicted Flight Price</div>
        <div style="font-size: 4rem; margin: 1rem 0;">
            Rs.{prediction:,.0f}
        </div>
        <div style="font-size: 1.2rem; opacity: 0.9;">
            95% Confidence: Rs.{lower_bound:,.0f} - Rs.{upper_bound:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Price Range</h4>
            <p>Rs.{lower_bound:,.0f} - Rs.{upper_bound:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if prediction < 5000:
            category, color = "Budget", "#28a745"
        elif prediction < 10000:
            category, color = "Economy", "#17a2b8"
        elif prediction < 15000:
            category, color = "Standard", "#ffc107"
        elif prediction < 20000:
            category, color = "Premium", "#fd7e14"
        else:
            category, color = "Luxury", "#dc3545"
        
        st.markdown(f"""
        <div class="metric-card" style="background: {color};">
            <h4>🏷️ Category</h4>
            <p>{category}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>✅ Confidence</h4>
            <p>94.14%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Flight details summary
    st.markdown("### 📋 Flight Details Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Route Information:**
        - **From:** {source}
        - **To:** {destination}
        - **Airline:** {airline}
        - **Stops:** {total_stops}
        - **Duration:** {duration_hours}h {duration_minutes}m
        """)
    
    with col2:
        st.markdown(f"""
        **Travel Details:**
        - **Date:** {journey_date.strftime('%d %B %Y')}
        - **Departure:** {dep_time.strftime('%H:%M')}
        - **Day:** {journey_date.strftime('%A')}
        - **Additional Info:** {additional_info}
        """)
    
    # Price comparison chart
    st.markdown("### 📊 Price Comparison")
    
    avg_prices = {
        'Your Flight': prediction,
        'Budget Flights': 4500,
        'Economy Flights': 7500,
        'Premium Flights': 15000
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(avg_prices.keys()),
            y=list(avg_prices.values()),
            marker_color=['#667eea', '#28a745', '#17a2b8', '#dc3545'],
            text=[f'Rs.{v:,.0f}' for v in avg_prices.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Your Flight vs Average Prices",
        xaxis_title="Category",
        yaxis_title="Price (Rs.)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Success message
    st.markdown("""
    <div class="success-box">
        <h4>✅ Prediction Successful!</h4>
        <p>The price estimate is based on historical data and current market trends. 
        Actual prices may vary based on demand, availability, and booking time.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: PERFORMANCE
# ============================================================================

def show_performance_page(metadata):
    """Display model performance dashboard"""
    
    st.markdown("## 📊 Model Performance Dashboard")
    
    predictions_df, comparison_df = load_performance_data()
    
    if predictions_df is None or comparison_df is None:
        st.error("❌ Performance data not available")
        return
    
    # Overview metrics
    st.markdown("### 🎯 Performance Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Test MAE",
            f"Rs.{metadata['performance_metrics']['test_mae']:,.0f}",
            delta=f"{metadata['performance_metrics']['test_mape']:.2f}% MAPE"
        )
    
    with col2:
        st.metric(
            "Test RMSE",
            f"Rs.{metadata['performance_metrics']['test_rmse']:,.0f}"
        )
    
    with col3:
        st.metric(
            "R² Score",
            f"{metadata['performance_metrics']['test_r2']:.4f}"
        )
    
    with col4:
        st.metric(
            "CV MAE",
            f"Rs.{metadata['performance_metrics']['cv_mae']:,.0f}",
            delta=f"±Rs.{metadata['performance_metrics']['cv_std']:,.0f}"
        )
    
    with col5:
        within_20_pct = (predictions_df['Percentage_Error'] <= 20).mean() * 100
        st.metric(
            "Accuracy",
            f"{within_20_pct:.1f}%",
            delta="Within ±20%"
        )
    
    st.markdown("---")
    
    # Model comparison
    st.markdown("### 🏆 Model Comparison")
    
    comparison_sorted = comparison_df.sort_values('Test_MAE')
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Test MAE', 'Test R²', 'Training Time', 'Overfitting Score')
    )
    
    fig.add_trace(
        go.Bar(x=comparison_sorted['Model'], y=comparison_sorted['Test_MAE'],
               name='MAE', marker_color='#667eea'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=comparison_sorted['Model'], y=comparison_sorted['Test_R2'],
               name='R²', marker_color='#28a745'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=comparison_sorted['Model'], y=comparison_sorted['Training_Time'],
               name='Time', marker_color='#ffc107'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=comparison_sorted['Model'], y=comparison_sorted['Overfit_Score'],
               name='Overfit', marker_color='#dc3545'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction analysis
    st.markdown("---")
    st.markdown("### 🎯 Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            predictions_df,
            x='Actual_Price',
            y='Predicted_Price',
            color='Absolute_Error',
            color_continuous_scale='Viridis',
            title='Actual vs Predicted Prices',
            template='plotly_dark'
        )
        
        fig.add_trace(
            go.Scatter(
                x=[predictions_df['Actual_Price'].min(), predictions_df['Actual_Price'].max()],
                y=[predictions_df['Actual_Price'].min(), predictions_df['Actual_Price'].max()],
                mode='lines',
                name='Perfect',
                line=dict(color='red', dash='dash')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=predictions_df['Actual_Price'] - predictions_df['Predicted_Price'],
            nbinsx=50,
            marker_color='#667eea'
        ))
        
        fig.update_layout(
            title='Error Distribution',
            xaxis_title='Error (Rs.)',
            yaxis_title='Frequency',
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Accuracy metrics
    st.markdown("---")
    st.markdown("### ✅ Prediction Accuracy")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        within_10 = (predictions_df['Percentage_Error'] <= 10).sum()
        within_10_pct = (within_10 / len(predictions_df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>Within ±10%</h4>
            <p>{within_10_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        within_20 = (predictions_df['Percentage_Error'] <= 20).sum()
        within_20_pct = (within_20 / len(predictions_df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>Within ±20%</h4>
            <p>{within_20_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        within_30 = (predictions_df['Percentage_Error'] <= 30).sum()
        within_30_pct = (within_30 / len(predictions_df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>Within ±30%</h4>
            <p>{within_30_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 4: FEATURE ANALYSIS
# ============================================================================

def show_feature_analysis_page(df):
    """Display feature analysis"""
    
    st.markdown("## 📈 Feature Analysis")
    
    feature_importance_df = load_feature_importance()
    
    if feature_importance_df is not None:
        st.markdown("### 🎯 Feature Importance")
        
        # Top 20 features
        top_20 = feature_importance_df.head(20)
        
        fig = go.Figure(data=[
            go.Bar(
                y=top_20['Feature'],
                x=top_20['Importance'],
                orientation='h',
                marker_color='#667eea'
            )
        ])
        
        fig.update_layout(
            title='Top 20 Most Important Features',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_dark',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.markdown("### 📋 Feature Importance Table")
        st.dataframe(
            feature_importance_df.head(30),
            use_container_width=True,
            height=400
        )
    
    if df is not None:
        st.markdown("---")
        st.markdown("### 📊 Feature Distributions")
        
        # Select feature to analyze
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['Price', 'Price_Log']]
        
        selected_feature = st.selectbox(
            "Select Feature to Analyze:",
            numeric_cols
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x=selected_feature,
                nbins=50,
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                title=f'{selected_feature} Distribution',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                y=selected_feature,
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(
                title=f'{selected_feature} Box Plot',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

def show_about_page():
    """Display about page"""
    
    st.markdown("## ℹ️ About This System")
    
    st.markdown("""
    <div class="info-box">
        <h3>✈️ Flight Price Prediction System</h3>
        <p>An advanced machine learning system for predicting flight prices with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Project Overview")
    
    st.markdown("""
    This system uses **XGBoost (Extreme Gradient Boosting)** algorithm to predict flight prices
    based on various features such as:
    
    - **Airline**: Different airlines have different pricing strategies
    - **Route**: Source and destination cities
    - **Date & Time**: Journey date, departure time, day of week
    - **Duration**: Total flight duration
    - **Stops**: Number of stops (non-stop, 1 stop, 2 stops, etc.)
    - **Additional Services**: Online booking, table booking, etc.
    
    The model achieves **94.14% accuracy (R² score)** with an average error of only **Rs.606**.
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Machine Learning:**
        - Algorithm: XGBoost Regressor
        - Training Samples: 8,369
        - Test Samples: 2,093
        - Features: 55
        - Cross-Validation: 5-fold
        """)
    
    with col2:
        st.markdown("""
        **Performance:**
        - Test MAE: Rs.605.73
        - Test RMSE: Rs.1,104.38
        - Test R²: 0.9414
        - Test MAPE: 7.03%
        - Predictions within ±20%: 92%
        """)
    
    st.markdown("---")
    st.markdown("### 📚 Technologies Used")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Frontend:**
        - Streamlit
        - Plotly
        - HTML/CSS
        """)
    
    with col2:
        st.markdown("""
        **Machine Learning:**
        - XGBoost
        - Scikit-learn
        - Pandas
        - NumPy
        """)
    
    with col3:
        st.markdown("""
        **Deployment:**
        - Python 3.8+
        - Joblib
        - Pickle
        """)
    
    # ============================================================================
    # FOOTER - DEVELOPER INFORMATION (WITH ICONS)
    # ============================================================================

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer Information")

    st.markdown("""
    <div class="success-box">
        <p><strong>👤 Developed by:</strong> Nour Eldeen Mohammed</p>
        <p><strong>🚀 Project:</strong> Flight Price Prediction System</p>
        <p><strong>📌 Version:</strong> 1.0</p>
        <p><strong>📅 Last Updated:</strong> January 2026</p>
        <p><strong>🎓 Institution:</strong> Epsilon AI - AI Internship Program</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📧 Connect With Me")

    # Social media buttons
    st.markdown("""
    <style>
    .social-btn {
        display: inline-block;
        padding: 12px 24px;
        margin: 10px 5px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .social-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .email-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .linkedin-btn {
        background: linear-gradient(135deg, #0077b5 0%, #00a0dc 100%);
        color: white;
    }

    .github-btn {
        background: linear-gradient(135deg, #333 0%, #555 100%);
        color: white;
    }
    </style>

    <div style="text-align: center; margin: 30px 0;">
        <a href="mailto:nourlouta@gmail.com" class="social-btn email-btn" target="_blank">
            📧 Email Me
        </a>
        <a href="https://www.linkedin.com/in/nour-eldeen-mohammed-mba-0b439721a/" 
        class="social-btn linkedin-btn" target="_blank">
            💼 LinkedIn Profile
        </a>
        <a href="https://github.com/NourLouta" class="social-btn github-btn" target="_blank">
            🐙 GitHub Portfolio
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Project highlights
    st.markdown("### 🎯 Project Highlights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Model Accuracy",
            value="93.4%",
            delta="R² Score"
        )

    with col2:
        st.metric(
            label="⚡ Prediction Speed",
            value="< 1s",
            delta="Real-time"
        )

    with col3:
        st.metric(
            label="🎯 MAE",
            value="₹624",
            delta="Low Error"
        )

    with col4:
        st.metric(
            label="📈 Features",
            value="45+",
            delta="Engineered"
        )

    st.markdown("---")

    # About section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; border-radius: 15px; color: white; margin: 20px 0;">
        <h3 style="margin-top: 0; color: white;">💡 About This Project</h3>
        <p style="font-size: 16px; line-height: 1.6;">
            This <strong>end-to-end machine learning system</strong> predicts flight prices using 
            advanced regression techniques including Random Forest, XGBoost, and LightGBM. 
            The project demonstrates expertise in:
        </p>
        <ul style="font-size: 15px; line-height: 1.8;">
            <li>✅ <strong>Data Science:</strong> Comprehensive EDA, feature engineering, and statistical analysis</li>
            <li>✅ <strong>Machine Learning:</strong> Model selection, hyperparameter tuning, and ensemble methods</li>
            <li>✅ <strong>MLOps:</strong> Model deployment, version control, and production-ready code</li>
            <li>✅ <strong>Web Development:</strong> Interactive Streamlit dashboard with real-time predictions</li>
        </ul>
        <p style="font-size: 14px; margin-bottom: 0;">
            🎓 <strong>Built as part of Epsilon AI Internship Program</strong> - 
            Showcasing practical application of data science in real-world scenarios.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Technical stack
    st.markdown("### 🛠️ Technical Stack")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("""
        **Data Science**
        - 🐼 Pandas
        - 🔢 NumPy
        - 📊 Matplotlib/Seaborn
        - 📈 Plotly
        """)

    with tech_col2:
        st.markdown("""
        **Machine Learning**
        - 🤖 Scikit-learn
        - 🌲 Random Forest
        - 🚀 XGBoost/LightGBM
        - 🎯 CatBoost
        """)

    with tech_col3:
        st.markdown("""
        **Deployment**
        - 🎈 Streamlit
        - 🐙 Git/GitHub
        - 📦 Joblib
        - ☁️ Streamlit Cloud
        """)

    st.markdown("---")

    # Feedback section
    st.markdown("### 💬 Feedback & Collaboration")

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; 
                border-left: 5px solid #667eea;">
        <p style="margin: 0; font-size: 15px;">
            <strong>🤝 Open to Collaboration:</strong> 
            I'm always interested in discussing data science projects, machine learning innovations, 
            and potential collaborations. Feel free to reach out!
        </p>
        <p style="margin-top: 10px; font-size: 15px;">
            <strong>📝 Feedback Welcome:</strong> 
            Your feedback helps me improve. If you have suggestions or found this project helpful, 
            please connect with me on LinkedIn or GitHub.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Copyright footer
    st.markdown("""
    <div style="text-align: center; padding: 15px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white; margin-top: 30px;">
        <p style="margin: 0; font-size: 14px;">
            © 2026 <strong>Nour Eldeen Mohammed</strong> | All Rights Reserved
        </p>
        <p style="margin: 5px 0 0 0; font-size: 13px; opacity: 0.9;">
            Built with ❤️ using Python, Streamlit & Machine Learning | 
            <a href="https://github.com/NourLouta" target="_blank" 
            style="color: white; text-decoration: underline;">
                View Source Code
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()