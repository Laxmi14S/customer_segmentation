# main_short.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Load Dataset & Rename Columns
# -----------------------------
df = pd.read_csv("data/Mall_Customers.csv")
df.rename(columns={'Annual Income (k$)': 'AnnualIncome',
                   'Spending Score (1-100)': 'SpendingScore'}, inplace=True)

# -----------------------------
# Preprocessing & Clustering
# -----------------------------
X = df[['AnnualIncome', 'SpendingScore']]
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42).fit(X_scaled)
df['Cluster'] = kmeans.labels_

# -----------------------------
# Assign Cluster Names
# -----------------------------
summary = df.groupby('Cluster')[['AnnualIncome','SpendingScore']].mean()
def name_cluster(row):
    inc, spend = row['AnnualIncome'], row['SpendingScore']
    if inc>70 and spend>70: return "Premium"
    elif inc<40 and spend>60: return "Impulsive"
    elif inc<40 and spend<40: return "Budget"
    elif inc>70 and spend<40: return "Careful"
    else: return "Average"
df['ClusterName'] = summary.apply(name_cluster, axis=1).reindex(df['Cluster']).values

# -----------------------------
# Combined Dashboard (2x2)
# -----------------------------
fig, axes = plt.subplots(2,2, figsize=(16,12))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='ClusterName',
                palette='Set2', data=df, s=120, alpha=0.8, ax=axes[0,0])
axes[0,0].set_title("Customer Segments")

metrics = {
    "Number of Customers": df['ClusterName'].value_counts(),
    "Average Spending Score": df.groupby('ClusterName')['SpendingScore'].mean(),
    "Average Annual Income": df.groupby('ClusterName')['AnnualIncome'].mean()
}

for ax, (title, data) in zip([axes[0,1], axes[1,0], axes[1,1]], metrics.items()):
    sns.barplot(x=data.index, y=data.values, palette='Set2', ax=ax)
    ax.set_title(title)
    ax.set_ylabel(title.split()[-1])
    ax.set_xlabel("Segment")
    if title=="Average Spending Score": ax.set_ylim(0,100)

plt.tight_layout()
plt.show()

# -----------------------------
# Print Insights
# -----------------------------
print(df.groupby('ClusterName')[['AnnualIncome','SpendingScore']].mean())
print(df['ClusterName'].value_counts())
