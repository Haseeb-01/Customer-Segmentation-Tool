import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from io import BytesIO

st.set_page_config(page_title="Customer Segmentation Tool", layout="wide")
st.title("📊 Customer Segmentation with K-Means Clustering")
st.markdown("""
Upload your dataset to explore customer segments using machine learning.
You'll receive cluster visualizations, summaries, and downloadable results.
""")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a CSV File", type=['csv'])

if uploaded_file:
    try:
        with st.spinner('Reading and processing file...'):
            data = pd.read_csv(uploaded_file)

        st.success("✅ File uploaded and loaded successfully!")
        st.subheader("🔍 Data Preview")
        st.dataframe(data.head())

        # Detect numeric features
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("🚫 Not enough numeric features for clustering.")
        else:
            # Feature Selection
            variances = data[numeric_cols].var().sort_values(ascending=False)
            default_features = variances.head(4).index.tolist()
            selected_features = st.multiselect("✅ Select features for clustering:", numeric_cols, default=default_features)

            if len(selected_features) < 2:
                st.warning("⚠️ Please select at least 2 features to proceed.")
            else:
                try:
                    with st.spinner('🔄 Preprocessing and running clustering...'):
                        # Prepare Data
                        X = data[selected_features]

                        # Handle missing values
                        if X.isnull().sum().sum() > 0:
                            st.warning("⚠️ Missing values detected. Filling with column medians.")
                            X = X.fillna(X.median())

                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        # Elbow Method
                        wcss = []
                        for i in range(1, 11):
                            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                            kmeans.fit(X_scaled)
                            wcss.append(kmeans.inertia_)

                        st.subheader("📈 Elbow Plot")
                        fig1, ax1 = plt.subplots()
                        ax1.plot(range(1, 11), wcss, marker='o')
                        ax1.set_title('Elbow Method')
                        ax1.set_xlabel('Number of Clusters')
                        ax1.set_ylabel('WCSS')
                        st.pyplot(fig1)

                        # Cluster slider
                        k = st.slider("Select number of clusters:", min_value=2, max_value=10, value=4)

                        # KMeans Model
                        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
                        clusters = kmeans.fit_predict(X_scaled)
                        data['Cluster'] = clusters

                        silhouette = silhouette_score(X_scaled, clusters)
                        st.info(f"Silhouette Score: {silhouette:.2f}")

                        # PCA & Scatter Plot
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        fig2, ax2 = plt.subplots()
                        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=80)
                        ax2.set_title("Clusters in 2D using PCA")
                        ax2.set_xlabel("PC1")
                        ax2.set_ylabel("PC2")
                        st.subheader("🔬 Cluster Visualization")
                        st.pyplot(fig2)

                        # Cluster Summary
                        st.subheader("📋 Cluster Summary")
                        summary = data.groupby('Cluster')[selected_features].mean().round(2)
                        st.dataframe(summary)

                        # Download Buttons
                        st.subheader("⬇️ Download Results")
                        csv_full = data.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Clustered Data (CSV)", csv_full, file_name='clustered_data.csv')

                        csv_summary = summary.to_csv().encode('utf-8')
                        st.download_button("Download Cluster Summary (CSV)", csv_summary, file_name='cluster_summary.csv')

                except Exception as cluster_error:
                    st.error(f"❌ An error occurred during clustering: {cluster_error}")

    except Exception as file_error:
        st.error(f"❌ Failed to read the CSV file. Make sure it's properly formatted.\nDetails: {file_error}")
else:
    st.info("📥 Please upload a CSV file to begin.")
    st.markdown("Or try [this sample dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv).")