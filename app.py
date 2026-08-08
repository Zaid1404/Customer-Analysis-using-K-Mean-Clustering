import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS Styling (Minimalist Deep Charcoal + Emerald Accent)
# ---------------------------------------------------------
custom_css = """
<style>
    /* Main container styling */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar custom styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }

    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #10B981;
        transform: translateY(-2px);
    }
    .metric-card h4 {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card p {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card span {
        font-size: 0.85rem;
        color: #10B981;
        font-weight: 500;
    }

    /* Primary Accent Button */
    .stButton>button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
    }

    /* Section headers */
    .section-header {
        border-bottom: 2px solid #334155;
        padding-bottom: 8px;
        margin-bottom: 20px;
        color: #38BDF8;
        font-weight: 600;
    }

    /* Streamlit native widget customization */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #CBD5E1 !important;
        font-size: 0.9rem !important;
    }

    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper: Data Loading & Processing
# ---------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "Mall_Customers.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Fallback synthetic dataset generation matching Mall_Customers structure
        np.random.seed(42)
        n_samples = 200
        df = pd.DataFrame({
            'CustomerID': range(1, n_samples + 1),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 70, n_samples),
            'Annual Income (k$)': np.random.randint(15, 137, n_samples),
            'Spending Score (1-100)': np.random.randint(1, 99, n_samples)
        })
    return df

def assign_business_label(income, spending):
    if income > 65 and spending > 60:
        return "VIP Customers"
    elif income > 65 and spending <= 60:
        return "High Income / Low Spending"
    elif income <= 45 and spending > 60:
        return "Low Income / High Spending"
    elif income <= 45 and spending <= 40:
        return "Budget Customers"
    else:
        return "Regular Customers"

# Map segments to UI badge colors
label_colors = {
    "VIP Customers": "#10B981",              # Emerald Green
    "High Income / Low Spending": "#3B82F6", # Blue
    "Low Income / High Spending": "#F59E0B", # Amber
    "Budget Customers": "#EF4444",           # Red
    "Regular Customers": "#8B5CF6"           # Purple
}

# ---------------------------------------------------------
# Load Data & Sidebar Setup
# ---------------------------------------------------------
df_raw = load_data()

st.sidebar.title("🛍️ Cluster Config")
st.sidebar.markdown("---")

# User Controls
selected_k = st.sidebar.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=5,
    help="K=5 is optimal according to the Elbow method."
)

show_raw_data = st.sidebar.checkbox("Show Raw Dataset", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Real-Time Predictor")
st.sidebar.caption("Classify a new customer instantly:")

input_income = st.sidebar.number_input("Annual Income (k$)", min_value=10, max_value=200, value=70, step=5)
input_spending = st.sidebar.slider("Spending Score (1-100)", min_value=1, max_value=100, value=75)
predict_btn = st.sidebar.button("Classify Customer")

# ---------------------------------------------------------
# Clustering Logic Execution
# ---------------------------------------------------------
X = df_raw[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df = df_raw.copy()
df['Cluster'] = clusters
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)

silhouette_avg = silhouette_score(X_scaled, clusters)

# Calculate Cluster Summary
summary_df = df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)
summary_df["Business Label"] = summary_df.apply(
    lambda r: assign_business_label(r["Annual Income (k$)"], r["Spending Score (1-100)"]), axis=1
)

# Map labels back to full dataset
cluster_label_map = summary_df["Business Label"].to_dict()
df["Segment"] = df["Cluster"].map(cluster_label_map)

# ---------------------------------------------------------
# Main Dashboard UI Layout
# ---------------------------------------------------------
st.title("📊 Customer Segmentation Analytics")
st.caption("AI-Powered K-Means Clustering for Retail Customer Insights")

st.markdown("<br>", unsafe_allow_html=True)

# Top KPI Metrics Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""<div class="metric-card">
            <h4>Total Customers</h4>
            <p>{len(df)}</p>
            <span>Active Database</span>
        </div>""",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""<div class="metric-card">
            <h4>Active Clusters (K)</h4>
            <p>{selected_k}</p>
            <span>Configured via Sidebar</span>
        </div>""",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""<div class="metric-card">
            <h4>Silhouette Score</h4>
            <p>{silhouette_avg:.3f}</p>
            <span>Cluster Quality Index</span>
        </div>""",
        unsafe_allow_html=True
    )

with col4:
    quality_text = "Excellent" if silhouette_avg > 0.5 else "Moderate" if silhouette_avg > 0.25 else "Weak"
    st.markdown(
        f"""<div class="metric-card">
            <h4>Separation Quality</h4>
            <p>{quality_text}</p>
            <span>Validation Result</span>
        </div>""",
        unsafe_allow_html=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Real-Time Prediction Modal / Alert Area
# ---------------------------------------------------------
if predict_btn:
    user_input_scaled = scaler.transform([[input_income, input_spending]])
    predicted_cluster = kmeans.predict(user_input_scaled)[0]
    predicted_label = assign_business_label(input_income, input_spending)
    color = label_colors.get(predicted_label, "#10B981")

    st.markdown("### 🔮 Classification Result")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.markdown(
            f"""
            <div style="background-color: #1E293B; border-left: 6px solid {color}; padding: 20px; border-radius: 8px;">
                <h4 style="margin:0; color:#94A3B8;">Target Segment</h4>
                <h2 style="margin:5px 0; color:{color};">{predicted_label}</h2>
                <p style="margin:0; color:#CBD5E1;">Assigned to Cluster #{predicted_cluster}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with res_col2:
        st.info(f"Customer with **${input_income}k** Annual Income and **{input_spending}/100** Spending Score aligns best with the **{predicted_label}** group.")
    st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Core Analytics Visualizations
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Customer Clusters", "📈 Optimal K (Elbow Method)", "📋 Segment Profiles"])

with tab1:
    st.markdown("<h3 class='section-header'>Customer Clusters & Centroids</h3>", unsafe_allow_html=True)
    
    # Plotly 2D Scatter Plot
    fig_clusters = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Segment",
        color_discrete_map=label_colors,
        hover_data=["CustomerID", "Age", "Gender"],
        template="plotly_dark",
        height=550
    )

    # Add Centroid Markers
    fig_clusters.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers',
            marker=dict(symbol='x', size=14, color='white', line=dict(width=2, color='black')),
            name='Centroids'
        )
    )

    fig_clusters.update_layout(
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        legend=dict(title="Customer Segment", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_clusters, use_container_width=True)

with tab2:
    st.markdown("<h3 class='section-header'>Elbow Method Inertia Curve</h3>", unsafe_allow_html=True)
    
    # Calculate Inertia over K=1..10
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_test.fit(X_scaled)
        inertia.append(km_test.inertia_)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(
        go.Scatter(
            x=list(K_range),
            y=inertia,
            mode='lines+markers',
            marker=dict(size=10, color='#10B981'),
            line=dict(color='#38BDF8', width=3)
        )
    )
    
    # Highlight chosen K
    fig_elbow.add_vline(x=selected_k, line_dash="dash", line_color="#EF4444", annotation_text=f" Current K={selected_k}")

    fig_elbow.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        height=500
    )

    st.plotly_chart(fig_elbow, use_container_width=True)

with tab3:
    st.markdown("<h3 class='section-header'>Cluster Persona & Summary</h3>", unsafe_allow_html=True)
    
    # Cluster counts
    counts = df["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Customer Count"]

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Customer Distribution")
        fig_pie = px.pie(
            counts,
            names="Segment",
            values="Customer Count",
            color="Segment",
            color_discrete_map=label_colors,
            hole=0.4,
            template="plotly_dark"
        )
        fig_pie.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### Cluster Averages")
        st.dataframe(
            summary_df.style.background_gradient(cmap="Blues"),
            use_container_width=True,
            height=300
        )

# ---------------------------------------------------------
# Raw Data View (Optional Toggle)
# ---------------------------------------------------------
if show_raw_data:
    st.markdown("---")
    st.markdown("### 📄 Processed Dataset")
    st.dataframe(df, use_container_width=True)