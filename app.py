import io
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation Analytics", page_icon="🛍️", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/hannazan/mall-customer-segmentation/main/data/Mall_Customers.csv"

st.title("🛍️ Customer Segmentation & Marketing Analytics")
st.caption("Unsupervised machine learning with K-Means clustering")

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_URL)

def normalize_columns(df):
    df = df.copy()
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"customerid", "customer id", "customer_id", "id"}:
            rename[col] = "CustomerID"
        elif key in {"gender", "genre", "sex"}:
            rename[col] = "Gender"
        elif key in {"age"}:
            rename[col] = "Age"
        elif key in {"annual income (k$)", "annual income", "annual_income", "income"}:
            rename[col] = "Annual Income (k$)"
        elif key in {"spending score (1-100)", "spending score", "spending_score", "spendingscore"}:
            rename[col] = "Spending Score (1-100)"
    return df.rename(columns=rename)

with st.sidebar:
    st.header("Data & model settings")
    uploaded = st.file_uploader("Upload a customer CSV (optional)", type=["csv"])
    k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)
    st.caption("Choose numeric behavior features. Customer ID is excluded from clustering.")

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source_label = "Uploaded CSV"
    else:
        raw = load_default_data()
        source_label = "Mall Customers dataset (GitHub)"
except Exception as e:
    st.error("Could not load the default dataset. Upload a CSV file using the sidebar.")
    st.caption(f"Load detail: {e}")
    st.stop()

df = normalize_columns(raw)
if df.empty:
    st.error("The uploaded dataset has no rows.")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_features = [c for c in numeric_cols if c.lower() not in {"customerid", "customer id", "customer_id", "id"}]
preferred = [c for c in ["Age", "Annual Income (k$)", "Spending Score (1-100)"] if c in numeric_features]
if not preferred:
    preferred = numeric_features

if len(numeric_features) < 2:
    st.error("Please upload a CSV with at least two numeric customer features, such as Age, Income, or Spending Score.")
    st.stop()

with st.sidebar:
    features = st.multiselect("Features used for clustering", options=numeric_features, default=preferred)
if len(features) < 2:
    st.warning("Select at least two numeric features to run clustering.")
    st.stop()

model_df = df[features].apply(pd.to_numeric, errors="coerce")
valid_mask = model_df.notna().all(axis=1)
clean = df.loc[valid_mask].copy()
model_df = model_df.loc[valid_mask]
if len(model_df) <= k:
    st.error(f"Need more valid customer rows than K={k}.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(model_df)
model = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)
clean["Cluster"] = labels.astype(int).astype(str)
sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 and len(labels) > k else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers analyzed", f"{len(clean):,}")
c2.metric("Clusters", str(k))
c3.metric("Silhouette score", "N/A" if np.isnan(sil) else f"{sil:.3f}")
c4.metric("Features selected", str(len(features)))
st.caption(f"Data source: {source_label}. Rows with missing/non-numeric selected features are excluded from clustering.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Cluster profiles", "Customer explorer", "Method & limitations"])

with tab1:
    left, right = st.columns(2)
    with left:
        counts = clean["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Customers")
        fig = px.bar(counts, x="Cluster", y="Customers", color="Cluster", title="Customers in each cluster",
                     text="Customers")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if len(features) == 2:
            plot_df = clean.copy()
            fig = px.scatter(plot_df, x=features[0], y=features[1], color="Cluster",
                             hover_data=[c for c in ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"] if c in plot_df.columns],
                             title="Customer clusters")
        else:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            plot_df = clean.copy()
            plot_df["PCA 1"], plot_df["PCA 2"] = coords[:, 0], coords[:, 1]
            fig = px.scatter(plot_df, x="PCA 1", y="PCA 2", color="Cluster",
                             title="Customer clusters (PCA 2D projection)",
                             hover_data=[c for c in features if c in plot_df.columns])
            fig.update_layout(xaxis_title="Principal component 1", yaxis_title="Principal component 2")
        st.plotly_chart(fig, use_container_width=True)
    st.info("Cluster numbers are identifiers, not rankings. The algorithm discovers groups; the marketing labels are interpretations based on each group's profile.")

with tab2:
    profile = clean.groupby("Cluster")[features].mean().round(2)
    profile.insert(0, "Customers", clean.groupby("Cluster").size())
    st.subheader("Average profile of each cluster")
    st.dataframe(profile, use_container_width=True)
    if "Annual Income (k$)" in features and "Spending Score (1-100)" in features:
        st.plotly_chart(px.scatter(clean, x="Annual Income (k$)", y="Spending Score (1-100)", color="Cluster",
                                   title="Income versus spending score"), use_container_width=True)
    st.subheader("Potential marketing interpretations")
    for cluster_id, row in profile.iterrows():
        means = row[features]
        high_spend = "Spending Score (1-100)" in features and means.get("Spending Score (1-100)", 0) >= clean["Spending Score (1-100)"].median() if "Spending Score (1-100)" in features else False
        high_income = "Annual Income (k$)" in features and means.get("Annual Income (k$)", 0) >= clean["Annual Income (k$)"].median() if "Annual Income (k$)" in features else False
        if high_spend and high_income:
            idea = "Consider loyalty benefits or early access to relevant offers."
        elif high_spend:
            idea = "Test personalized recommendations or bundles while monitoring profitability."
        elif high_income:
            idea = "Test premium options only if purchase behavior supports the offer."
        else:
            idea = "Consider value-focused campaigns and measure response before scaling."
        st.write(f"**Cluster {cluster_id}:** {idea}")

with tab3:
    st.subheader("Explore assigned customers")
    cluster_choice = st.selectbox("Select cluster", sorted(clean["Cluster"].unique(), key=int))
    view = clean[clean["Cluster"] == cluster_choice]
    st.write(f"Customers in selected cluster: **{len(view)}**")
    st.dataframe(view, use_container_width=True)
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("Download this cluster as CSV", csv, file_name=f"cluster_{cluster_choice}_customers.csv", mime="text/csv")

with tab4:
    st.markdown("""
    **Workflow**
    1. Select numeric customer features and remove rows missing selected values.
    2. Standardize features with `StandardScaler` so variables on larger scales do not dominate distances.
    3. Fit K-Means with the selected number of clusters.
    4. Review cluster profiles, visualizations, and the silhouette score.

    **Interpretation and limitations**
    - This is unsupervised learning: there are no known correct segment labels, so ordinary classification accuracy is not appropriate.
    - Use the Elbow method and silhouette score as guides, not as automatic proof that a clustering is best.
    - The Mall Customers dataset has a small number of demographic/spending fields and does not contain transaction-level purchase history.
    - Marketing suggestions are hypotheses. A business should test campaigns and measure outcomes before acting on them.
    - Cluster numbers are arbitrary; cluster 0 is not inherently better than cluster 1.
    """)
