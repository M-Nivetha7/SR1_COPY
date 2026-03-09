# 🌌 SDSS SkyServer ML Classification

A machine learning project to classify celestial objects (Stars, Galaxies, and Quasars) using Random Forest Classifier.

## 🚀 Quick Start

Run these commands in order:

bash
# 1. Navigate to project directory
```
cd /Users/nivetham/Documents/space-astronomy-ml
```

# 2. Activate virtual environment
```
source venv/bin/activate
```

# 3. Generate ML model results
```
cd analysis
python3 realistic_model.py
```

# 4. Start the web server
```
cd ../website
python3 -m http.server 8000
```
🌐 View Dashboard
Open your browser and go to:

text
```
http://localhost:8000
```

📊 Model Performance
Training Accuracy: 86.69%

Testing Accuracy: 85.00%

Generalization Gap: 1.69%

🔧 Troubleshooting
Port already in use
bash
```
lsof -ti:8000 | xargs kill -9
python3 -m http.server 8000
```
Missing packages
bash
```
pip install pandas numpy scikit-learn
```
📁 Project Structure

```
space-astronomy-ml/
├── analysis/
│   └── realistic_model.py
├── website/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── results.json
└── README.md
```
