import pandas as pd
import numpy as np

# Create sample SDSS-like data
np.random.seed(42)
n_samples = 10000

# Generate features
data = {
    'objid': np.random.randint(1.23e18, 1.24e18, n_samples),
    'ra': np.random.uniform(8, 260, n_samples),
    'dec': np.random.uniform(-5, 68, n_samples),
    'u': np.random.uniform(12, 20, n_samples),
    'g': np.random.uniform(12, 20, n_samples),
    'r': np.random.uniform(12, 25, n_samples),
    'i': np.random.uniform(11, 28, n_samples),
    'z': np.random.uniform(11, 23, n_samples),
    'run': np.random.randint(300, 1413, n_samples),
    'rerun': 301,
    'camcol': np.random.randint(1, 7, n_samples),
    'field': np.random.randint(10, 769, n_samples),
    'specobjid': np.random.uniform(2.99e17, 9.47e18, n_samples),
    'redshift': np.random.uniform(-0.004, 5.35, n_samples),
}

# Create classes with realistic distribution
classes = np.random.choice(['STAR', 'GALAXY', 'QSO'], n_samples, p=[0.42, 0.5, 0.08])
data['class'] = classes

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("Skyserver_SQL2_27_2018 6_51_39 PM.csv", index=False)
print("✅ Sample dataset created successfully!")
print(f"📊 Shape: {df.shape}")
print(f"📈 Class distribution:")
print(df['class'].value_counts())
