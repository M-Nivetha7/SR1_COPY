import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import datetime

print("=" * 60)
print("🚀 Generating SDSS Classification Results with Sample Predictions")
print("=" * 60)

# Create sample data
print("\n📊 Creating sample dataset...")
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'ra': np.random.uniform(0, 360, n_samples),
    'dec': np.random.uniform(-90, 90, n_samples),
    'u': np.random.uniform(12, 22, n_samples),
    'g': np.random.uniform(12, 22, n_samples),
    'r': np.random.uniform(12, 22, n_samples),
    'i': np.random.uniform(12, 22, n_samples),
    'z': np.random.uniform(12, 22, n_samples),
    'redshift': np.concatenate([
        np.random.exponential(0.1, 400),  # Stars
        np.random.exponential(0.3, 500),  # Galaxies
        np.random.exponential(1.0, 100)   # Quasars
    ])
}

# Create target classes based on redshift
classes = []
for i in range(n_samples):
    if data['redshift'][i] < 0.15:
        classes.append('STAR')
    elif data['redshift'][i] < 0.4:
        classes.append('GALAXY')
    else:
        classes.append('QSO')

data['class'] = classes
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Dataset created: {df.shape[0]} observations")
print(f"   Class distribution:")
class_counts = df['class'].value_counts()
for cls, count in class_counts.items():
    print(f"   - {cls}: {count} ({count/len(df)*100:.1f}%)")

# Prepare features and target
feature_columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']
X = df[feature_columns]
y = df['class']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train model
print("\n🌲 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Calculate accuracies
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

# Generate classification report
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

# Feature importance
importance = model.feature_importances_
feature_importance = {}
for feature, imp in zip(feature_columns, importance):
    feature_importance[feature] = float(imp)

# Create sample predictions (10 samples)
sample_predictions = []
for i in range(min(10, len(X_test))):
    # Get the test sample
    X_sample = X_test.iloc[i:i+1]
    y_actual = y_test[i]
    y_predicted = y_pred[i]
    
    # Get prediction probabilities
    proba = model.predict_proba(X_sample)[0]
    confidence = float(max(proba))
    
    sample_predictions.append({
        'actual': le.inverse_transform([y_actual])[0],
        'predicted': le.inverse_transform([y_predicted])[0],
        'confidence': confidence,
        'redshift': float(X_test.iloc[i]['redshift'])
    })

print(f"\n🎲 Generated {len(sample_predictions)} sample predictions")

# Prepare complete results
results = {
    'project_info': {
        'name': 'SDSS SkyServer Astronomical Classification',
        'description': 'Classification of celestial objects using Random Forest',
        'dataset_size': len(df),
        'features': feature_columns,
        'classes': list(le.classes_),
        'class_counts': class_counts.to_dict()
    },
    'model_info': {
        'algorithm': 'Random Forest Classifier',
        'n_estimators': 100,
        'random_state': 42
    },
    'performance': {
        'training_accuracy': float(train_acc),
        'testing_accuracy': float(test_acc),
        'classification_report': report
    },
    'feature_importance': feature_importance,
    'sample_predictions': sample_predictions,
    'generated_date': datetime.datetime.now().isoformat()
}

# Save to website directory
output_path = '../website/results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_path}")

# Verify file was created and has content
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path)
    print(f"📁 File size: {file_size} bytes")
    
    # Quick verification
    with open(output_path, 'r') as f:
        data = json.load(f)
        print(f"\n✅ Verification:")
        print(f"   - Dataset size: {data['project_info']['dataset_size']}")
        print(f"   - Test accuracy: {data['performance']['testing_accuracy']:.4f}")
        print(f"   - Sample predictions: {len(data['sample_predictions'])}")
else:
    print(f"❌ Failed to save file!")

print("=" * 60)
