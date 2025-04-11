import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("diabetes_project.csv")

# Step 1: Data Pre-processing
# 1.1 Remove outliers using IQR
def remove_outliers_iqr(df):
    df_cleaned = df.copy()
    for col in df.columns:
        if df[col].dtype != 'O':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

df_no_outliers = remove_outliers_iqr(df)

# 1.2 Impute missing values with median
df_imputed = df_no_outliers.fillna(df_no_outliers.median(numeric_only=True))

# 1.3 Normalize all columns
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

# Step 2: Unsupervised Learning for Generating Labels
# 2.1 Select features for clustering
clustering_features = df_normalized[['Glucose', 'BMI', 'Age']]

# 2.2 Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(clustering_features)

# 2.3 Determine cluster for 'Diabetes'
cluster_centers = kmeans.cluster_centers_
diabetes_cluster = np.argmax(cluster_centers[:, 0])

# 2.4 Assign labels
df_normalized['Outcome'] = (clusters == diabetes_cluster).astype(int)

# Step 3: Feature Extraction with PCA
# 3.1 Separate features and target
X = df_normalized.drop(columns=['Outcome'])
y = df_normalized['Outcome']

# 3.2 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.3 Apply PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Convert to DataFrames for clarity
X_train_pca_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3'])
X_test_pca_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2', 'PC3'])

# X_train_pca_df and y_train to train your classification model (super learner).
print(X_train_pca_df.head())
print(y_train.head())
