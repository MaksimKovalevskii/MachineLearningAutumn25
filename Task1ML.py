import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import random
import sklearn
print(sklearn.__version__)
np.random.seed(42)
random.seed(42)

# Load dataset
data, meta = arff.loadarff('dataset.arff')
df = pd.DataFrame(data)

print(f"Dataset shape: {df.shape}")

# Convert bytes to strings
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Prepare data
target_col = df.columns[-1]
y = df[target_col]
X = df.drop([target_col], axis=1)

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

X = X.fillna(X.mean(numeric_only=True))

# Encode target
y_encoded = y.values

print(f"Features: {X.columns.tolist()}")
print(f"Total features: {len(X.columns)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#we use X_scaled initially for PCA and t-SNE, training and test set later for model predictions
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

#PCA on full dataset
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA Variance - PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}")

#Here I checked different perplexities for t-SNE
#tsne2 = TSNE(n_components=2, random_state=42,perplexity=100, verbose=1)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', alpha=0.6, s=15)
axes[0].set_title('PCA Full dataset')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
plt.colorbar(scatter1, ax=axes[0])

scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap='viridis', alpha=0.6, s=15)
axes[1].set_title('t-SNE Full dataset')
axes[1].set_xlabel('Component 1')
axes[1].set_ylabel('Component 2')
plt.colorbar(scatter2, ax=axes[1])

plt.tight_layout()
plt.savefig('dimensionality_reduction.png', dpi=300)
plt.close()

# PCA Biplot
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', alpha=0.3, s=15)
for i, feature in enumerate(X.columns):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, fontsize=9)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA Biplot: Features + Data Points')
plt.savefig('pca_biplot.png', dpi=300)
plt.close()

# 14 variables t-SNE (1-6 first plot, 7-14 second)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feature in enumerate(X.columns[:6]):
    ax = axes[idx]

    # Color points by feature values
    scatter = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=X[feature],
        cmap='viridis', alpha=0.6, s=10
    )

    ax.set_title(f't-SNE colored by: {feature}')
    plt.colorbar(scatter, ax=ax, label=feature)

#plt.savefig('tsne_features_exploration.png', dpi=300, bbox_inches='tight')
plt.close()

# Get remaining features (7 to end)
remaining_features = X.columns[6:]

# Calculate grid size
n_features = len(remaining_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
axes = axes.flatten()

for idx, feature in enumerate(remaining_features):
    ax = axes[idx]

    scatter = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=X[feature],
        cmap='viridis', alpha=0.6, s=10
    )

    ax.set_title(f't-SNE colored by: {feature}', fontweight='bold')

    plt.colorbar(scatter, ax=ax, label=feature)

# Hide extra subplots
for idx in range(len(remaining_features), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('tsne_features_7_14.png', dpi=300, bbox_inches='tight')
plt.close()

#Predictions comparison stage - RandomForest
#Forest with all varaibles (14)
model_original = RandomForestRegressor(n_estimators=100, random_state=42)
model_original.fit(X_train, y_train)

y_pred_original = model_original.predict(X_test)
r2_original = r2_score(y_test, y_pred_original)
mae_original = mean_absolute_error(y_test, y_pred_original)
rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))

print(f"  R² score: {r2_original:.4f}")
print(f"  MAE: {mae_original:.2f}")
print(f"  RMSE: {rmse_original:.2f}")

#Forest with PCA
pca_model = PCA(n_components=2)
X_train_pca = pca_model.fit_transform(X_train)
X_test_pca = pca_model.transform(X_test)

model_pca = RandomForestRegressor(n_estimators=100, random_state=42)
model_pca.fit(X_train_pca, y_train)

y_pred_pca = model_pca.predict(X_test_pca)
r2_pca = r2_score(y_test, y_pred_pca)
mae_pca = mean_absolute_error(y_test, y_pred_pca)
rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))

print(f"  R² score: {r2_pca:.4f}")
print(f"  MAE: {mae_pca:.2f}")
print(f"  RMSE: {rmse_pca:.2f}")

#Forest with t-SNE
tsne_model = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne_model.fit_transform(X_train)

model_tsne = RandomForestRegressor(n_estimators=100, random_state=42)
model_tsne.fit(X_train_tsne, y_train)

y_pred_tsne = model_tsne.predict(X_train_tsne)
r2_tsne = r2_score(y_train, y_pred_tsne)

print(f"  R² score (train only): {r2_tsne:.4f}")
print(f"  ALL METHODS ")
results_all = pd.DataFrame({
    'Method': ['Original (14 features)', 'PCA (2 components)', 't-SNE (2 components)'],
    'R²': [f"{r2_original:.4f}", f"{r2_pca:.4f}", f"{r2_tsne:.4f} (train)"],
    'MAE': [f"{mae_original:.2f}", f"{mae_pca:.2f}", "N/A"],
    'RMSE': [f"{rmse_original:.2f}", f"{rmse_pca:.2f}", "N/A"]
})

print(results_all.to_string(index=False))

#Here all the same but we exclude casual and registered features
# ============================================================================
X = X.drop(['casual', 'registered'], axis=1)

print(f"Features after removal: {X.columns.tolist()}")
print(f"Total features: {len(X.columns)}")

# splitting data set again
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


model_original = RandomForestRegressor(n_estimators=100, random_state=42)
model_original.fit(X_train, y_train)

y_pred_original = model_original.predict(X_test)
r2_original = r2_score(y_test, y_pred_original)
mae_original = mean_absolute_error(y_test, y_pred_original)
rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))

print(f"  R² score: {r2_original:.4f}")
print(f"  MAE: {mae_original:.2f}")
print(f"  RMSE: {rmse_original:.2f}")

pca_model = PCA(n_components=2)
X_train_pca = pca_model.fit_transform(X_train)
X_test_pca = pca_model.transform(X_test)

model_pca = RandomForestRegressor(n_estimators=100, random_state=42)
model_pca.fit(X_train_pca, y_train)

y_pred_pca = model_pca.predict(X_test_pca)
r2_pca = r2_score(y_test, y_pred_pca)
mae_pca = mean_absolute_error(y_test, y_pred_pca)
rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))

print(f"  R² score: {r2_pca:.4f}")
print(f"  MAE: {mae_pca:.2f}")
print(f"  RMSE: {rmse_pca:.2f}")


tsne_model = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne_model.fit_transform(X_train)

model_tsne = RandomForestRegressor(n_estimators=100, random_state=42)
model_tsne.fit(X_train_tsne, y_train)

y_pred_tsne = model_tsne.predict(X_train_tsne)
r2_tsne = r2_score(y_train, y_pred_tsne)

print(f"  R² score (train only): {r2_tsne:.4f}")

print(f"  ALL METHODS - 2 vars excluded ")

results_clean = pd.DataFrame({
    'Method': ['Original (12 features)', 'PCA (2 components)', 't-SNE (2 components)'],
    'R²': [f"{r2_original:.4f}", f"{r2_pca:.4f}", f"{r2_tsne:.4f} (train)"],
    'MAE': [f"{mae_original:.2f}", f"{mae_pca:.2f}", "N/A"],
    'RMSE': [f"{rmse_original:.2f}", f"{rmse_pca:.2f}", "N/A"]
})

print(results_clean.to_string(index=False))
