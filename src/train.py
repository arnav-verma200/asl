import numpy as np
import os
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
X_train = np.load('data/processed/X_train.npy', allow_pickle=True)
X_val   = np.load('data/processed/X_val.npy',   allow_pickle=True)
y_train = np.load('data/processed/y_train.npy', allow_pickle=True)
y_val   = np.load('data/processed/y_val.npy',   allow_pickle=True)

print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"Data loaded successfully!")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean',
        n_jobs=-1
    ),
    'MLP Neural Network': MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=100,
        random_state=42
    )
}

os.makedirs('models', exist_ok=True)
results = {}

print("\n" + "="*50)
print("TRAINING ALL MODELS")
print("="*50)

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time   = time.time()

    train_time = round(end_time - start_time, 2)

    y_pred    = model.predict(X_val)
    accuracy  = accuracy_score(y_val, y_pred)
    accuracy_percent = round(accuracy * 100, 2)

    results[model_name] = {
        'accuracy': accuracy_percent,
        'time': train_time,
        'model': model
    }

    print(f"✅ {model_name}")
    print(f"   Accuracy:      {accuracy_percent}%")
    print(f"   Training time: {train_time} seconds")

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"\n{'Model':<25} {'Accuracy':>10} {'Time':>10}")
print("-"*50)

for model_name, result in results.items():
    print(f"{model_name:<25} {result['accuracy']:>9}% {result['time']:>9}s")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model      = results[best_model_name]['model']
best_accuracy   = results[best_model_name]['accuracy']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"🎯 Accuracy:   {best_accuracy}%")

for model_name, result in results.items():
    filename = model_name.lower().replace(' ', '_') + '_model.pkl'
    joblib.dump(result['model'], f'models/{filename}')
    print(f"💾 Saved: models/{filename}")

joblib.dump(best_model, 'models/best_model.pkl')
print(f"\n✅ Best model saved as models/best_model.pkl")
print("Training complete!")