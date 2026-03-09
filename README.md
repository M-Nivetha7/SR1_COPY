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
![WhatsApp Image 2026-03-09 at 16 53 19](https://github.com/user-attachments/assets/d7a63f08-0fae-4621-a672-083b56c04327)
![WhatsApp Image 2026-03-09 at 16 53 19 (1)](https://github.com/user-attachments/assets/af9eb1b4-b562-4dd3-850a-8520ce0cc488)
![WhatsApp Image 2026-03-09 at 16 53 19 (2)](https://github.com/user-attachments/assets/5ee2ad89-262c-4267-b8ec-545abbc93eb3)
![WhatsApp Image 2026-03-09 at 16 53 19 (3)](https://github.com/user-attachments/assets/0f54fa4c-25c5-414e-9c61-728dee226f42)




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
