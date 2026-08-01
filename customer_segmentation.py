
# Customer Segmentation using K-Means Clustering
# Dataset: Mall_Customers.csv (place in same folder)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("="*60)
print("CUSTOMER SEGMENTATION USING K-MEANS")
print("="*60)

df = pd.read_csv("Mall_Customers.csv")

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows")
print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

print("\nSummary Statistics")
print(df.describe())

# Feature Selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertia = []
K = range(1,11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.show()

print("\nFrom the elbow graph, K=5 is commonly chosen for this dataset.\n")

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

score = silhouette_score(X_scaled, clusters)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)

print("="*60)
print("MODEL EVALUATION")
print("="*60)
print(f"Number of Clusters : {kmeans.n_clusters}")
print(f"Silhouette Score   : {score:.3f}")

if score > 0.7:
    print("Excellent cluster separation.")
elif score > 0.5:
    print("Good clustering quality.")
elif score > 0.25:
    print("Reasonable clustering.")
else:
    print("Weak clustering.")

print("\nCluster Sizes")
print(df["Cluster"].value_counts().sort_index())

summary = df.groupby("Cluster")[["Age","Annual Income (k$)","Spending Score (1-100)"]].mean().round(2)
print("\nAverage values per Cluster")
print(summary)

labels = []
for _, r in summary.iterrows():
    inc = r["Annual Income (k$)"]
    spend = r["Spending Score (1-100)"]

    if inc > 65 and spend > 60:
        labels.append("VIP Customers")
    elif inc > 65 and spend <= 60:
        labels.append("High Income / Low Spending")
    elif inc <= 45 and spend > 60:
        labels.append("Low Income / High Spending")
    elif inc <= 45 and spend <= 40:
        labels.append("Budget Customers")
    else:
        labels.append("Regular Customers")

summary["Business Label"] = labels

print("\nBusiness Interpretation")
print(summary)

plt.figure(figsize=(8,6))

for c in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"]==c]
    plt.scatter(
        subset["Annual Income (k$)"],
        subset["Spending Score (1-100)"],
        label=f"Cluster {c}"
    )

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker="X",
    s=250,
    linewidths=2,
    label="Centroids"
)

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("customer_clusters.png")
plt.show()

df.to_csv("customers_with_clusters.csv", index=False)

print("\nFiles Generated:")
print("- elbow_method.png")
print("- customer_clusters.png")
print("- customers_with_clusters.csv")

print("\nProject Completed Successfully!")
