# 🎯 Gamble Comment Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.68+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

<p align="center">
  <em>An intelligent machine learning system for detecting gambling-related content in user comments using natural language processing techniques.</em>
</p>

---

## ✨ Features

- **🤖 Smart Detection**: Advanced ML model trained to identify gambling-related comments with high accuracy
- **⚡ Fast API**: Lightning-fast REST API built with FastAPI for real-time inference
- **📊 Model Evaluation**: Comprehensive evaluation metrics and performance monitoring
- **🔄 Easy Integration**: Simple API endpoints for seamless integration into existing systems
- **📈 Scalable Architecture**: Modular design supporting easy model updates and improvements
- **🛠️ Development Tools**: Complete pipeline with training, evaluation, and inference capabilities

---

## 🚀 Quick Start

### Prerequisites

Ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/minggo-commits/gamble-comment-detector.git
   cd gamble-comment-detector
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 🔥 Start the API Server
```bash
python app.py
```

The API will be available at `http://localhost:8000`

#### 📝 API Documentation
Once the server is running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

#### 🧪 Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your comment text here"}'
```

#### 🏋️ Train the Model
```bash
python src/train.py
```

#### 📊 Evaluate Model Performance
```bash
python src/evaluate.py
```

---

## 🏗️ Project Structure

```
gamble-comment-detector/
├── 📁 .github/
│   └── workflows/          # CI/CD pipeline configurations
├── 📁 app/
│   ├── __init__.py
│   └── main.py            # FastAPI application
├── 📁 data/
│   └── comments.csv       # Training dataset
├── 📁 model/
│   ├── eval_report.json   # Model evaluation metrics
│   ├── saved_model.joblib # Trained ML model
│   └── vectorizer.joblib  # Text vectorizer
├── 📁 notebooks/
│   └── Baseline.ipynb     # Jupyter notebook for experimentation
├── 📁 scripts/
│   └── generate_dummy_data.py  # Data generation utilities
├── 📁 src/
│   ├── __init__.py
│   ├── evaluate.py        # Model evaluation
│   ├── inference.py       # Prediction logic
│   ├── preprocessing.py   # Data preprocessing
│   └── train.py          # Model training
├── app.py                # Main application entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Analyze a single comment |
| `POST` | `/predict/batch` | Analyze multiple comments |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/model/info` | Get model information |

### Example API Usage

**Single Prediction:**
```json
POST /predict
{
    "text": "I just won big at the casino last night!"
}

Response:
{
    "is_gambling": true,
    "confidence": 0.87,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 🧠 Model Information

- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Features**: Text preprocessing with stop word removal and stemming
- **Performance**: 
  - Accuracy: 92%
  - Precision: 89%
  - Recall: 94%
  - F1-Score: 91%

---

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black src/ app/
flake8 src/ app/
```

### Generate Dummy Data
```bash
python scripts/generate_dummy_data.py
```

---

## 📈 Model Training Pipeline

1. **Data Preprocessing** (`src/preprocessing.py`)
   - Text cleaning and normalization
   - Feature extraction using TF-IDF

2. **Model Training** (`src/train.py`)
   - Train machine learning model
   - Hyperparameter optimization
   - Model serialization

3. **Evaluation** (`src/evaluate.py`)
   - Performance metrics calculation
   - Cross-validation
   - Generate evaluation reports

4. **Deployment** (`app/main.py`)
   - FastAPI application setup
   - Model loading and inference

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📋 Roadmap

- [x] ✅ Basic ML model implementation
- [x] ✅ FastAPI REST API
- [x] ✅ Model evaluation metrics
- [ ] 🔄 Real-time model retraining
- [ ] 🔄 Advanced deep learning models
- [ ] 🔄 Multi-language support
- [ ] 🔄 Model monitoring dashboard
- [ ] 🔄 Docker containerization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to all contributors who helped build this project
- Special recognition to the open-source ML community
- Built with ❤️ using FastAPI, scikit-learn, and Python

---

## 📞 Support

- 🐛 **Found a bug?** [Open an issue](https://github.com/minggo-commits/gamble-comment-detector/issues)
- 💡 **Have a feature request?** [Start a discussion](https://github.com/minggo-commits/gamble-comment-detector/discussions)
- 📧 **Need help?** Contact the maintainers

---

<p align="center">
  <strong>⭐ Don't forget to star this repository if you found it helpful!</strong>
</p>