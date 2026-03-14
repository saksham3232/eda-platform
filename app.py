# ============================================================================
# EXPLORATORY DATA ANALYSIS PLATFORM WITH ML PREPROCESSING
# ENHANCED VERSION: New features + Filters default to EMPTY
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EDA Platform with ML Preprocessing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME & CUSTOM CSS
# ============================================================================
def get_theme_css(dark_mode):
    if dark_mode:
        return """
        <style>
        /* ── DARK MODE: full app override ── */
        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stApp"], .stApp {
            background-color: #0e1117 !important;
            color: #e6edf3 !important;
        }
        /* Main content area */
        [data-testid="stMainBlockContainer"],
        [data-testid="block-container"],
        .main .block-container {
            background-color: #0e1117 !important;
            color: #e6edf3 !important;
        }
        /* Sidebar */
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background-color: #161b22 !important;
            color: #e6edf3 !important;
        }
        /* All text */
        h1, h2, h3, h4, h5, h6, p, span, label,
        .stMarkdown, .stText, div[data-testid="stMarkdownContainer"] {
            color: #e6edf3 !important;
        }
        /* Metrics */
        [data-testid="stMetric"] label,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #c9d1d9 !important;
        }
        /* Input widgets */
        .stTextInput input, .stSelectbox div,
        .stMultiSelect div[data-baseweb="select"] {
            background-color: #21262d !important;
            color: #e6edf3 !important;
            border-color: #30363d !important;
        }
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #161b22 !important;
            gap: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            color: #8b949e !important;
        }
        .stTabs [aria-selected="true"] {
            color: #58a6ff !important;
            border-bottom-color: #58a6ff !important;
        }
        /* Dataframe / table */
        [data-testid="stDataFrame"] iframe {
            filter: invert(0.9) hue-rotate(180deg);
        }
        /* Expander */
        details, .streamlit-expanderHeader {
            background-color: #161b22 !important;
            color: #e6edf3 !important;
        }
        /* Divider */
        hr { border-color: #30363d !important; }
        /* Custom cards */
        .metric-card {
            background-color: #161b22 !important;
            padding: 1rem; border-radius: 10px;
            border-left: 5px solid #58a6ff;
            color: #e6edf3 !important;
        }
        .ml-pipeline-card {
            background-color: #1a2e1a !important;
            padding: 1rem; border-radius: 10px;
            border-left: 5px solid #3fb950;
        }
        .filter-info {
            background-color: #1c2128 !important;
            border: 1px solid #30363d;
            border-radius: 8px; padding: 0.6rem 1rem;
            font-size: 0.85rem; color: #8b949e !important;
        }
        </style>
        """
    else:
        return """
        <style>
        /* ── LIGHT MODE: restore defaults ── */
        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stApp"], .stApp {
            background-color: #ffffff !important;
            color: #31333f !important;
        }
        [data-testid="stMainBlockContainer"],
        [data-testid="block-container"],
        .main .block-container {
            background-color: #ffffff !important;
            color: #31333f !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background-color: #f0f2f6 !important;
            color: #31333f !important;
        }
        h1,h2,h3,h4,h5,h6,p,span,label,
        .stMarkdown,.stText,div[data-testid="stMarkdownContainer"] {
            color: #31333f !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
        /* Custom cards */
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem; border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .ml-pipeline-card {
            background-color: #e8f5e8;
            padding: 1rem; border-radius: 10px;
            border-left: 5px solid #28a745;
        }
        .filter-info {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px; padding: 0.6rem 1rem;
            font-size: 0.85rem; color: #6c757d;
        }
        </style>
        """

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
defaults = {
    'uploaded_file': None, 'df': None,
    'needs_preprocessing': True, 'preprocessed_df': None,
    'filter_values': {}, 'dark_mode': False,
    'scaler_choice': 'StandardScaler',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# ML PREPROCESSING FUNCTIONS
# ============================================================================
def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = pd.DataFrame(num_imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = pd.DataFrame(cat_imputer.fit_transform(df[categorical_cols]), columns=categorical_cols, index=df.index)
    return df

def apply_feature_scaling(df, scaler_type='StandardScaler'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        if scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler_type == 'RobustScaler':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        df[numeric_cols] = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    return df

def apply_encoding(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def preprocess_for_ml(df, scaler_type='StandardScaler'):
    df_processed = handle_missing_values(df.copy())
    df_processed = apply_feature_scaling(df_processed, scaler_type)
    df_processed = apply_encoding(df_processed)
    return df_processed

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def detect_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def detect_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns.tolist()

def calculate_summary_statistics(df, numeric_cols):
    stats_data = {
        'Column': [], 'Count': [], 'Mean': [], 'Median': [],
        'Std Dev': [], 'Min': [], 'Max': [], '25th %': [], '75th %': []
    }
    for col in numeric_cols:
        stats_data['Column'].append(col)
        stats_data['Count'].append(df[col].count())
        stats_data['Mean'].append(f"{df[col].mean():.2f}")
        stats_data['Median'].append(f"{df[col].median():.2f}")
        stats_data['Std Dev'].append(f"{df[col].std():.2f}")
        stats_data['Min'].append(f"{df[col].min():.2f}")
        stats_data['Max'].append(f"{df[col].max():.2f}")
        stats_data['25th %'].append(f"{df[col].quantile(0.25):.2f}")
        stats_data['75th %'].append(f"{df[col].quantile(0.75):.2f}")
    return pd.DataFrame(stats_data)


def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] < lower) | (df[col] > upper)], lower, upper

def create_histogram(df, column, bins=30):
    fig = px.histogram(df, x=column, nbins=bins, title=f'📈 Distribution of {column}',
                      color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_title=column, yaxis_title='Frequency', height=450, showlegend=False)
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None):
    fig = px.scatter(df, x=x_col, y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=f'🔗 {x_col} vs {y_col}', height=500, opacity=0.7)
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(hovermode='closest')
    return fig

def create_bar_chart(df, x_col, y_col, title=None):
    try:
        grouped_data = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False)
        fig = px.bar(grouped_data, x=x_col, y=y_col, title=title or f'📊 {y_col} by {x_col}',
                    color=y_col, color_continuous_scale='Viridis')
    except:
        fig = px.bar(df, x=x_col, y=y_col, title=f'📊 {y_col} by {x_col}', color=y_col)
    fig.update_layout(height=450)
    return fig

def create_line_chart(df, x_col, y_col, color_col=None):
    color_param = color_col if color_col and color_col in df.columns else None
    fig = px.line(df, x=x_col, y=y_col, color=color_param,
                 title=f'📈 {y_col} over {x_col}', markers=True, height=450)
    return fig

def create_box_plot(df, x_col, y_col):
    fig = px.box(df, x=x_col, y=y_col, title=f'📦 Box Plot: {y_col} by {x_col}', height=450, notched=True)
    return fig

def create_pie_chart(df, values_col, names_col, title=None):
    try:
        grouped_data = df.groupby(names_col)[values_col].sum().reset_index()
        fig = px.pie(grouped_data, values=values_col, names=names_col,
                    title=title or f'🥧 {values_col} Distribution by {names_col}', height=500, hole=0.3)
    except:
        fig = px.pie(df, values=values_col, names=names_col, title=f'🥧 {values_col} Distribution', hole=0.3)
    return fig

def create_violin_plot(df, x_col, y_col):
    fig = px.violin(df, x=x_col, y=y_col, box=True, points="outliers",
                    title=f'🎻 Violin: {y_col} by {x_col}', height=450)
    return fig

def create_correlation_heatmap(df, numeric_cols):
    corr_matrix = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale='RdBu_r', zmid=0, text=np.round(corr_matrix.values, 2),
        texttemplate='%{text:.2f}', textfont={"size": 12}, hoverongaps=False
    ))
    fig.update_layout(title='🔥 Correlation Heatmap', height=550, xaxis_title='', yaxis_title='')
    return fig

# ============================================================================
# REUSABLE FILTER BUILDER — default=[] (NOTHING selected)
# ============================================================================
def build_categorical_filters(df, categorical_cols, key_prefix):
    """
    Returns a filtered DataFrame.
    Filters default to EMPTY — user must choose values manually.
    Shows a clear info banner when no filter is active.
    """
    filtered_df = df.copy()

    if not categorical_cols:
        return filtered_df

    with st.expander("🎯 **Categorical Filters** — select values to narrow data", expanded=True):
        st.markdown(
            '<div class="filter-info">💡 No filter selected = all rows shown. Pick values below to filter.</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        filter_cols = st.columns(min(len(categorical_cols), 4))
        active_filters = 0

        for idx, col in enumerate(categorical_cols):
            with filter_cols[idx % len(filter_cols)]:
                unique_vals = sorted(df[col].dropna().unique().tolist())
                selected = st.multiselect(
                    f"**{col}** ({len(unique_vals)} unique)",
                    options=unique_vals,
                    default=[],          # ← EMPTY BY DEFAULT
                    key=f"{key_prefix}_filter_{col}",
                    placeholder="All values (unfiltered)"
                )
                if selected:
                    filtered_df = filtered_df[filtered_df[col].isin(selected)]
                    active_filters += 1

        total = len(df)
        after = len(filtered_df)
        pct = after / total * 100 if total > 0 else 100
        if active_filters > 0:
            st.success(f"✅ **{active_filters} filter(s) active** — showing **{after:,}** of **{total:,}** rows ({pct:.1f}%)")
        else:
            st.info(f"ℹ️ No filters active — showing all **{total:,}** rows")

    return filtered_df


def build_numeric_range_filters(df, numeric_cols, key_prefix):
    """Optional numeric range filters."""
    filtered_df = df.copy()
    if not numeric_cols:
        return filtered_df

    with st.expander("🔢 **Numeric Range Filters** (optional)", expanded=False):
        selected_num_cols = st.multiselect(
            "Add range filter for columns:",
            numeric_cols,
            default=[],
            key=f"{key_prefix}_num_filter_select"
        )
        for col in selected_num_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            lo, hi = st.slider(
                f"**{col}** range",
                min_value=col_min, max_value=col_max,
                value=(col_min, col_max),
                key=f"{key_prefix}_range_{col}"
            )
            filtered_df = filtered_df[(filtered_df[col] >= lo) & (filtered_df[col] <= hi)]

    return filtered_df





# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # ── Theme toggle — must happen FIRST, before any st.stop() ───────────────
    with st.sidebar:
        st.markdown("## ⚙️ **Settings**")
        dark_toggle = st.toggle(
            "🌙 Dark Mode",
            value=st.session_state.dark_mode,
            key="dark_toggle"
        )
        if dark_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_toggle
            st.rerun()
        st.markdown("---")

    # Inject theme CSS immediately — before anything else renders
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)


    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 📊 Exploratory Data Analysis Platform with **ML Preprocessing**")
    st.markdown("### **Advanced data exploration + ML-ready data pipeline**")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Upload any CSV/Excel dataset and unlock:**")
        st.markdown("- 🔍 **Smart filters** (default empty — you choose what to filter)")
        st.markdown("- 🔢 **Numeric range sliders** for precise row filtering")
        st.markdown("- 📈 **7+ interactive chart types** including Violin plots")
        st.markdown("- 🤖 **ML Pipeline** — choose scaler + full preprocessing export")
        st.markdown("- 🌙 **Dark mode** toggle")
    with col2:
        st.info("✨ **Enhanced v2**\n• Empty filters by default\n• Violin charts\n• Outlier detection\n• Scaler selection")

    # ── Sidebar: Upload ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📁 **Data Upload**")
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload for instant analysis"
        )

        if uploaded_file is not None:
            prev_name = st.session_state.get("uploaded_file_name", None)
            if prev_name != uploaded_file.name:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    df_raw = pd.read_csv(uploaded_file) if file_extension == 'csv' else pd.read_excel(uploaded_file)
                    st.session_state.df = df_raw
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.needs_preprocessing = True
                    st.session_state.preprocessed_df = None
                except Exception as e:
                    st.error(f"❌ **Upload Error**: {str(e)}")
                    st.stop()
            st.success(f"✅ **Loaded!**\n{len(st.session_state.df):,} rows × {len(st.session_state.df.columns)} cols")
        else:
            st.warning("📤 **Upload a file** to begin analysis")
            st.stop()

    if st.session_state.df is None:
        st.stop()

    df = st.session_state.df
    numeric_cols = detect_numeric_columns(df)
    categorical_cols = detect_categorical_columns(df)

    if not numeric_cols:
        st.error("❌ **No numeric columns found!**")
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 **Overview**", "🔍 **Exploration**", "📈 **Visualizations**",
        "🔬 **Advanced**", "📋 **Data**", "🤖 **ML Pipeline**"
    ])

    # =========================================================================
    # TAB 1: OVERVIEW
    # =========================================================================
    with tab1:
        st.markdown("## **Dataset Overview & Quick Insights**")
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("Total Rows", f"{len(df):,}"),
            ("Total Columns", len(df.columns)),
            ("Numeric Columns", len(numeric_cols)),
            ("Categorical Columns", len(categorical_cols)),
        ]
        for c, (label, val) in zip([col1, col2, col3, col4], metrics):
            with c:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(label, val)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Column Information**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Missing': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(1),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Summary Statistics**")
            summary_df = calculate_summary_statistics(df, numeric_cols)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("### **Missing Values Analysis**")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        if len(missing_data) > 0:
            fig_missing = px.bar(
                missing_data.head(10), x='Column', y='Missing %',
                title='Missing Values Distribution',
                color='Missing %', color_continuous_scale='Reds', height=400
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ **Perfect! No missing values detected**")

        # Data type distribution pie
        st.markdown("### **Column Type Breakdown**")
        type_counts = df.dtypes.astype(str).value_counts().reset_index()
        type_counts.columns = ['dtype', 'count']
        fig_types = px.pie(type_counts, values='count', names='dtype',
                           title='Column Data Types', hole=0.4, height=350)
        st.plotly_chart(fig_types, use_container_width=True)

    # =========================================================================
    # TAB 2: EXPLORATION
    # =========================================================================
    with tab2:
        st.markdown("## 🔍 **Interactive Data Exploration**")

        base_df = df.copy()

        # Categorical filters — EMPTY BY DEFAULT
        filtered_df = build_categorical_filters(base_df, categorical_cols, key_prefix="explore")
        # Numeric range filters
        filtered_df = build_numeric_range_filters(filtered_df, numeric_cols, key_prefix="explore")

        current_rows = len(filtered_df)
        if current_rows > 1:
            row_start, row_end = st.slider(
                "Select row range", 0, current_rows - 1,
                (0, min(49, current_rows - 1)), key="explore_row_range"
            )
            slice_df = filtered_df.iloc[row_start:row_end + 1]
        else:
            slice_df = filtered_df
            row_start, row_end = 0, max(0, current_rows - 1)

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Filtered Rows", f"{len(filtered_df):,}")
        with col2: st.metric("Original Rows", f"{len(df):,}")
        with col3:
            pct = len(filtered_df) / len(df) * 100 if len(df) > 0 else 0
            st.metric("Retained %", f"{pct:.1f}%")
        with col4: st.metric("Displaying", f"{len(slice_df):,}")

        if numeric_cols and len(filtered_df) > 0:
            numeric_col = st.selectbox("Select metric for quick stats:", numeric_cols, key="explore_metric")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Mean", f"{filtered_df[numeric_col].mean():.2f}")
            c2.metric("Median", f"{filtered_df[numeric_col].median():.2f}")
            c3.metric("Max", f"{filtered_df[numeric_col].max():.2f}")
            c4.metric("Min", f"{filtered_df[numeric_col].min():.2f}")
            c5.metric("Std Dev", f"{filtered_df[numeric_col].std():.2f}")
            c6.metric("Skewness", f"{filtered_df[numeric_col].skew():.2f}")

            col_a, col_b = st.columns(2)
            with col_a:
                fig = create_histogram(filtered_df, numeric_col)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig_box = px.box(filtered_df, y=numeric_col,
                                 title=f"📦 Box Plot: {numeric_col}", height=450)
                st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("### **Filtered Data Preview**")
        st.dataframe(slice_df, use_container_width=True)

    # =========================================================================
    # TAB 3: VISUALIZATIONS
    # =========================================================================
    with tab3:
        st.markdown("## **📈 Interactive Visualizations**")

        base_df = df.copy()
        viz_df = build_categorical_filters(base_df, categorical_cols, key_prefix="viz")
        viz_df = build_numeric_range_filters(viz_df, numeric_cols, key_prefix="viz")

        v_max = len(viz_df) - 1
        if v_max > 0:
            v_start, v_end = st.slider(
                "Row range for chart", 0, v_max,
                (0, min(v_max, 1500)), key="viz_row_range"
            )
            viz_df = viz_df.iloc[v_start:v_end + 1]

        chart_type = st.selectbox(
            "🎨 **Chart Type**",
            ["Histogram", "Scatter Plot", "Bar Chart", "Line Chart",
             "Box Plot", "Violin Plot", "Pie Chart"],
            key="viz_chart_type"
        )

        col1, col2 = st.columns(2)

        if chart_type == "Histogram" and numeric_cols:
            with col1: x_col = st.selectbox("Column:", numeric_cols, key="hist_col")
            bins = st.slider("Number of bins:", 5, 100, 30, key="hist_bins")
            if len(viz_df) > 0:
                fig = create_histogram(viz_df, x_col, bins)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
            with col1: x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
            with col2: y_col = st.selectbox("Y-axis:", [c for c in numeric_cols if c != x_col], key="scatter_y")
            color_options = [None] + categorical_cols
            color_col = st.selectbox("Color by:", color_options, key="scatter_color")
            if len(viz_df) > 0:
                fig = create_scatter_plot(viz_df, x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart" and numeric_cols:
            with col1: x_col = st.selectbox("X-axis:", categorical_cols + numeric_cols, key="bar_x")
            with col2: y_col = st.selectbox("Y-axis:", numeric_cols, key="bar_y")
            if len(viz_df) > 0:
                fig = create_bar_chart(viz_df, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line Chart" and numeric_cols:
            with col1: x_col = st.selectbox("X-axis:", df.columns.tolist(), key="line_x")
            with col2: y_col = st.selectbox("Y-axis:", numeric_cols, key="line_y")
            color_col = st.selectbox("Color by:", [None] + categorical_cols, key="line_color")
            if len(viz_df) > 0:
                fig = create_line_chart(viz_df, x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot" and numeric_cols:
            with col1: x_col = st.selectbox("X-axis (group):", categorical_cols + numeric_cols, key="box_x")
            with col2: y_col = st.selectbox("Y-axis (value):", numeric_cols, key="box_y")
            if len(viz_df) > 0:
                fig = create_box_plot(viz_df, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Violin Plot" and numeric_cols:
            with col1: x_col = st.selectbox("X-axis (group):", categorical_cols + numeric_cols, key="violin_x")
            with col2: y_col = st.selectbox("Y-axis (value):", numeric_cols, key="violin_y")
            if len(viz_df) > 0:
                fig = create_violin_plot(viz_df, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie Chart" and numeric_cols:
            with col1: names_col = st.selectbox("Categories:", categorical_cols + numeric_cols, key="pie_names")
            with col2: values_col = st.selectbox("Values:", numeric_cols, key="pie_values")
            if len(viz_df) > 0:
                fig = create_pie_chart(viz_df, values_col, names_col)
                st.plotly_chart(fig, use_container_width=True)

        if len(viz_df) == 0:
            st.warning("⚠️ No data matches the current filters.")

    # =========================================================================
    # TAB 4: ADVANCED ANALYTICS
    # =========================================================================
    with tab4:
        st.markdown("## **🔬 Advanced Analytics**")

        analysis_df = df.copy()
        if len(analysis_df) > 2500:
            start, end = st.slider(
                "Analysis Sample Size", 0, len(analysis_df) - 1,
                (0, min(2499, len(analysis_df) - 1)), key="analysis_rows"
            )
            analysis_df = analysis_df.iloc[start:end + 1]

        analysis_type = st.selectbox(
            "🎯 **Analysis Type**",
            ["Correlation Analysis", "Distribution Analysis", "Comparative Analysis",
             "Outlier Analysis"],
            key="analysis_type"
        )
        st.markdown("---")

        if analysis_type == "Correlation Analysis" and len(numeric_cols) >= 2:
            st.markdown("### **🔥 Correlation Matrix**")
            fig = create_correlation_heatmap(analysis_df, numeric_cols)
            st.plotly_chart(fig, use_container_width=True)

            corr_matrix = analysis_df[numeric_cols].corr()
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Pair': f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                        'Correlation': round(corr_matrix.iloc[i, j], 3)
                    })
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🏆 Strongest Correlations**")
                st.dataframe(corr_df.head(5), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**📉 Weakest Correlations**")
                st.dataframe(corr_df.tail(5), use_container_width=True, hide_index=True)

        elif analysis_type == "Distribution Analysis" and numeric_cols:
            st.markdown("### **📊 Distribution Analysis**")
            selected_cols = st.multiselect(
                "Select columns:", numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))], key="dist_cols"
            )
            if selected_cols:
                for i, col in enumerate(selected_cols):
                    if i % 2 == 0:
                        cols = st.columns(2)
                    with cols[i % 2]:
                        fig = create_histogram(analysis_df, col)
                        st.plotly_chart(fig, use_container_width=True)
                st.markdown("### **📈 Distribution Summary**")
                st.dataframe(calculate_summary_statistics(analysis_df, selected_cols),
                             use_container_width=True, hide_index=True)

        elif analysis_type == "Comparative Analysis" and categorical_cols and numeric_cols:
            st.markdown("### **🔄 Comparative Analysis**")
            c1, c2 = st.columns(2)
            with c1: group_col = st.selectbox("Group by:", categorical_cols, key="comp_group")
            with c2: metric_col = st.selectbox("Metric:", numeric_cols, key="comp_metric")

            comp_data = analysis_df.groupby(group_col)[metric_col].agg(
                ['mean', 'sum', 'count', 'std']).round(2).reset_index()
            comp_data.columns = ['Group', 'Average', 'Total', 'Count', 'Std Dev']

            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.bar(comp_data, x='Group', y='Average',
                              title=f"Average {metric_col} by {group_col}",
                              color='Average', color_continuous_scale='Viridis', height=400)
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = px.bar(comp_data, x='Group', y='Total',
                              title=f"Total {metric_col} by {group_col}",
                              color='Total', color_continuous_scale='Plasma', height=400)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### **📋 Comparison Table**")
            st.dataframe(comp_data, use_container_width=True, hide_index=True)
            top = comp_data.loc[comp_data['Average'].idxmax(), 'Group']
            st.success(f"**🏆 Highest Average**: {top} ({comp_data['Average'].max():.2f})")

        elif analysis_type == "Outlier Analysis" and numeric_cols:
            st.markdown("### **🚨 Outlier Detection (IQR Method)**")
            outlier_col = st.selectbox("Select column:", numeric_cols, key="outlier_col")
            outliers_df, lower, upper = detect_outliers_iqr(analysis_df, outlier_col)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Outliers", len(outliers_df))
            c2.metric("Lower Bound", f"{lower:.2f}")
            c3.metric("Upper Bound", f"{upper:.2f}")

            fig_out = px.box(analysis_df, y=outlier_col, points="outliers",
                             title=f"Outliers in {outlier_col}", height=400)
            fig_out.add_hline(y=lower, line_dash="dash", line_color="red",
                              annotation_text="Lower fence")
            fig_out.add_hline(y=upper, line_dash="dash", line_color="red",
                              annotation_text="Upper fence")
            st.plotly_chart(fig_out, use_container_width=True)

            if len(outliers_df) > 0:
                st.markdown("**Outlier Rows:**")
                st.dataframe(outliers_df[[outlier_col] + categorical_cols[:3]].head(50),
                             use_container_width=True)
            else:
                st.success("✅ No outliers detected in this column.")

        else:
            st.info("⚠️ Select analysis type with sufficient data columns")

    # =========================================================================
    # TAB 5: RAW DATA
    # =========================================================================
    with tab5:
        st.markdown("## **📋 Raw Data Explorer**")

        base_df = df.copy()
        raw_df = build_categorical_filters(base_df, categorical_cols, key_prefix="raw")
        raw_df = build_numeric_range_filters(raw_df, numeric_cols, key_prefix="raw")

        col1, col2, col3 = st.columns(3)
        with col1: rows_to_show = st.slider("Rows to display:", 10, min(500, max(10, len(raw_df))), 25, key="raw_rows")
        with col2: sort_col = st.selectbox("Sort by:", raw_df.columns.tolist(), key="raw_sort")
        with col3: sort_asc = st.radio("Order:", ["Ascending", "Descending"], key="raw_order") == "Ascending"

        display_df = raw_df.sort_values(sort_col, ascending=sort_asc).head(rows_to_show)
        st.markdown(f"**Showing {len(display_df):,} of {len(raw_df):,} rows**")
        st.dataframe(display_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### **💾 Export Filtered Data**")
        c1, c2, c3 = st.columns(3)
        with c1:
            csv_data = raw_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv_data,
                               f"eda_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv", use_container_width=True)
        with c2:
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                raw_df.to_excel(writer, index=False, sheet_name='EDA')
            st.download_button("📊 Download Excel", out.getvalue(),
                               f"eda_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with c3:
            st.metric("📊 Filtered Rows", f"{len(raw_df):,}")

    # =========================================================================
    # TAB 6: ML PIPELINE
    # =========================================================================
    with tab6:
        st.markdown("## 🤖 **ML Preprocessing Pipeline**")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="ml-pipeline-card">', unsafe_allow_html=True)
            if st.session_state.needs_preprocessing:
                st.warning("🔄 Configure options and click **Run ML Pipeline**")
            else:
                st.success("✅ Preprocessing Complete! Data is ML-ready.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            scaler_choice = st.selectbox(
                "⚖️ Scaler",
                ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                help="StandardScaler: z-score | MinMax: [0,1] | Robust: outlier-resistant",
                key="scaler_select"
            )
            st.session_state.scaler_choice = scaler_choice

        col_run, col_reset = st.columns(2)
        with col_run:
            run_clicked = st.button("🚀 Run ML Pipeline", type="primary", use_container_width=True)
        with col_reset:
            reset_clicked = st.button("🗑️ Reset Pipeline", type="secondary", use_container_width=True)

        # Handle Reset
        if reset_clicked:
            st.session_state.needs_preprocessing = True
            st.session_state.preprocessed_df = None
            st.rerun()

        # Handle Run
        if run_clicked:
            with st.spinner("⏳ Running preprocessing pipeline…"):
                try:
                    result = preprocess_for_ml(df, scaler_choice)
                    st.session_state.preprocessed_df = result
                    st.session_state.needs_preprocessing = False
                    st.session_state.scaler_choice = scaler_choice
                    st.success("🎉 Pipeline complete! Scroll down to download.")
                except Exception as e:
                    st.error(f"❌ Pipeline failed: {str(e)}")

        # Show results if preprocessed data exists
        if st.session_state.preprocessed_df is not None:
            preprocessed_df = st.session_state.preprocessed_df

            st.success(f"✅ **Pipeline complete!** {len(preprocessed_df):,} rows × {len(preprocessed_df.columns)} columns are ML-ready.")
            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            c1.metric("Original Shape", f"{len(df):,} × {len(df.columns)}")
            c2.metric("Preprocessed Shape", f"{len(preprocessed_df):,} × {len(preprocessed_df.columns)}")
            c3.metric("Scaler Used", st.session_state.scaler_choice)

            st.markdown("**✅ Pipeline Steps Applied:**")
            st.markdown("- 🔧 **Missing Values**: Median (numeric), Mode (categorical)")
            st.markdown(f"- ⚖️ **Feature Scaling**: {st.session_state.scaler_choice}")
            st.markdown("- 🔤 **Encoding**: LabelEncoder (categorical → numeric)")

            st.markdown("---")
            st.markdown("### **💾 Download ML-Ready Data**")
            d1, d2 = st.columns(2)
            with d1:
                csv_ml = preprocessed_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Preprocessed CSV",
                    data=csv_ml,
                    file_name=f"ml_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="ml_csv_download"
                )
            with d2:
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    preprocessed_df.to_excel(writer, index=False, sheet_name="MLReady")
                st.download_button(
                    label="📊 Download Preprocessed Excel",
                    data=out.getvalue(),
                    file_name=f"ml_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="ml_excel_download"
                )

            st.markdown("### **👀 Preview: ML-Ready Data (first 10 rows)**")
            st.dataframe(preprocessed_df.head(10), use_container_width=True)

            st.markdown("### **📊 Preprocessed Statistics**")
            preprocessed_numeric = detect_numeric_columns(preprocessed_df)
            if preprocessed_numeric:
                st.dataframe(
                    calculate_summary_statistics(preprocessed_df, preprocessed_numeric),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("👆 Select your scaler above and click **Run ML Pipeline** to begin.")


if __name__ == "__main__":
    main()
