
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

data, meta = arff.loadarff('mnist_784.arff')
X = np.array(data.tolist(), dtype=np.float32)

# Separate features and labels
y = X[:, -1].astype(int)
X = X[:, :-1]
print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(" Data normalized")
print()

# We use MiniSom for Self-Organizing Map
som = MiniSom(x=10, y=10, input_len=784, sigma=3.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(X)
# Train
som.train_random(X, num_iteration=500, verbose=True)

print(" Training complete!")
print()

#Mapping samples to neurons
neuron_labels = [[[] for _ in range(10)] for _ in range(10)]

for i, sample in enumerate(X):
    winner = som.winner(sample)
    neuron_labels[winner[0]][winner[1]].append(y[i])

print(" Mapping complete!")
print()

dominant_digit = np.zeros((10, 10))
sample_count = np.zeros((10, 10))
purity = np.zeros((10, 10))

for i in range(10):
    for j in range(10):
        if len(neuron_labels[i][j]) > 0:
            labels = neuron_labels[i][j]
            sample_count[i, j] = len(labels)

            unique, counts = np.unique(labels, return_counts=True)
            dominant_digit[i, j] = unique[np.argmax(counts)]
            purity[i, j] = np.max(counts) / len(labels) * 100
        else:
            dominant_digit[i, j] = -1

# Plot 1: Dominant Digits
fig1, ax1 = plt.subplots(figsize=(10, 10))
cmap = plt.cm.get_cmap('tab10', 10)

for i in range(10):
    for j in range(10):
        if dominant_digit[i, j] >= 0:
            color = cmap(int(dominant_digit[i, j]))
            rect = plt.Rectangle((j, 10 - i - 1), 1, 1,
                                 facecolor=color, edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect)

            ax1.text(j + 0.5, 10 - i - 0.5,
                     f"{int(dominant_digit[i, j])}\n({int(sample_count[i, j])})",
                     ha='center', va='center', fontsize=8, fontweight='bold')

ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_aspect('equal')
ax1.set_title('SOM: Dominant Digit per Neuron', fontsize=14, fontweight='bold')
ax1.set_xlabel('Grid X')
ax1.set_ylabel('Grid Y')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=9))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, ticks=range(10))

plt.tight_layout()
plt.savefig('1_som_dominant_digits.png', dpi=150, bbox_inches='tight')
print("Saved: 1_som_dominant_digits.png")

# Plot 2: Sample Count
fig2, ax2 = plt.subplots(figsize=(10, 8))
im = ax2.imshow(sample_count, cmap='YlOrRd', interpolation='nearest')

for i in range(10):
    for j in range(10):
        ax2.text(j, i, int(sample_count[i, j]),
                 ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax2.set_title('SOM: Sample Count per Neuron', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax2, label='Count')
plt.tight_layout()
plt.savefig('2_som_sample_count.png', dpi=150, bbox_inches='tight')
print("Saved: 2_som_sample_count.png")

# Plot 3: Purity
fig3, ax3 = plt.subplots(figsize=(10, 8))
im = ax3.imshow(purity, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=100)

for i in range(10):
    for j in range(10):
        if purity[i, j] > 0:
            ax3.text(j, i, f"{purity[i, j]:.0f}%",
                     ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax3.set_title('SOM: Cluster Purity per Neuron (%)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3, label='Purity (%)')
plt.tight_layout()
plt.savefig('3_som_purity.png', dpi=150, bbox_inches='tight')
print("Saved: 3_som_purity.png")

# Plot 4: Weight Patterns
fig4, axes = plt.subplots(10, 10, figsize=(15, 15))
weights = som.get_weights()
for i in range(10):
    for j in range(10):
        ax = axes[i, j]
        # Get weight vector and reshape to 28x28
        weight = weights[i, j].reshape(28, 28)
        ax.imshow(weight, cmap='gray')
        ax.axis('off')

        if dominant_digit[i, j] >= 0:
            ax.set_title(f"{int(dominant_digit[i, j])}", fontsize=10, color='red', fontweight='bold')

fig4.suptitle('SOM: Learned Weight Patterns (28x28 pixels)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('4_som_weight_patterns.png', dpi=150, bbox_inches='tight')
print("Saved: 4_som_weight_patterns.png")


print("ANALYSIS SUMMARY")

total_neurons = 100
active_neurons = np.sum(sample_count > 0)
dead_neurons = total_neurons - active_neurons
avg_purity = np.mean(purity[purity > 0])

print(f"Total neurons: {total_neurons}")
print(f"Active neurons: {int(active_neurons)}")
print(f"Dead neurons: {int(dead_neurons)}")
print(f"Average purity: {avg_purity:.1f}%")
print()

print("Dominant digit distribution:")
for digit in range(10):
    count = np.sum(dominant_digit == digit)
    if count > 0:
        print(f"  Digit {digit}: {int(count)} neurons")

