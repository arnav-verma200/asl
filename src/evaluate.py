import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Load test data
print("Loading test data...")
X_test = np.load('data/processed/X_test.npy', allow_pickle=True)
y_test = np.load('data/processed/y_test.npy', allow_pickle=True)
print(f"X_test shape: {X_test.shape}")
print("Test data loaded!")

# Load all models
print("\nLoading all models...")
models = {
    'Random Forest':      joblib.load('models/random_forest_model.pkl'),
    'SVM':                joblib.load('models/svm_model.pkl'),
    'KNN':                joblib.load('models/knn_model.pkl'),
    'MLP Neural Network': joblib.load('models/mlp_neural_network_model.pkl')
}
print("All models loaded!")

# Evaluate all models
results = {}
print("\n" + "="*50)
print("EVALUATION ON TEST SET")
print("="*50)

for model_name, model in models.items():
    y_pred           = model.predict(X_test)
    accuracy         = accuracy_score(y_test, y_pred)
    accuracy_percent = round(accuracy * 100, 2)
    results[model_name] = {
        'accuracy': accuracy_percent,
        'y_pred':   y_pred
    }
    print(f"\n{model_name}: {accuracy_percent}%")

# Results table
print("\n" + "="*50)
print("FINAL RESULTS TABLE")
print("="*50)
print(f"\n{'Model':<25} {'Val Accuracy':>12} {'Test Accuracy':>13}")
print("-"*55)

val_accuracies = {
    'Random Forest':      97.94,
    'SVM':                99.07,
    'KNN':                95.48,
    'MLP Neural Network': 96.26
}

for model_name, result in results.items():
    val_acc  = val_accuracies[model_name]
    test_acc = result['accuracy']
    print(f"{model_name:<25} {val_acc:>11}% {test_acc:>12}%")

# Detailed report
best_model      = joblib.load('models/best_model.pkl')
y_pred_best     = best_model.predict(X_test)

print("\n" + "="*50)
print("DETAILED REPORT — BEST MODEL (SVM)")
print("="*50)
report = classification_report(y_test, y_pred_best)
print(report)

# Confusion matrix
os.makedirs('data/processed', exist_ok=True)
cm     = confusion_matrix(y_test, y_pred_best)
labels = sorted(list(set(y_test)))

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5
)
plt.title('Confusion Matrix — SVM Model', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('True Label', fontsize=13)
plt.tight_layout()
plt.savefig('data/processed/confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved!")

# Model comparison chart
model_names = list(results.keys())
test_accs   = [results[m]['accuracy'] for m in model_names]
val_accs    = [val_accuracies[m] for m in model_names]

x     = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, val_accs,  width, label='Validation Accuracy', color='steelblue',  alpha=0.8)
bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy',       color='darkorange', alpha=0.8)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() - 1.5,
            f'{bar.get_height()}%',
            ha='center', va='top',
            color='white', fontweight='bold', fontsize=10)

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() - 1.5,
            f'{bar.get_height()}%',
            ha='center', va='top',
            color='white', fontweight='bold', fontsize=10)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy %', fontsize=12)
ax.set_title('Model Comparison — Validation vs Test Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(85, 101)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/model_comparison.png', dpi=150)
plt.show()
print("Model comparison chart saved!")
print("\nEvaluation complete!")