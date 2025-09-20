# Gamble Comment Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.68+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/MLOps-Automated-orange.svg" alt="MLOps">
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen.svg" alt="GitHub Actions">
  <img src="https://img.shields.io/badge/Deploy-HuggingFace-yellow.svg" alt="Hugging Face">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

<p align="center">
  <em>An intelligent machine learning system with automated MLOps pipeline for detecting gambling-related content in user comments using natural language processing techniques.</em>
</p>

---

## Automated ML Pipeline (MLOps)

This project implements a complete **MLOps pipeline** using GitHub Actions that automatically handles the entire machine learning lifecycle:

### Pipeline Stages

| Stage | Description | Tools Used |
|-------|-------------|------------|
| Environment Setup | Python 3.9, dependencies installation | `setup-python@v4`, `pip` |
| Data Management | Auto-generate dummy data if missing | Custom Python script |
| Model Training | Train ML model with latest data | `scikit-learn`, `joblib` |
| Model Evaluation | Generate performance metrics and reports | Custom evaluation script |
| Artifact Management | Version control for model files | Git commits, `upload-artifact@v4` |
| Auto Deployment | Deploy to Hugging Face Spaces | Git LFS, HF Spaces API |

### Pipeline Triggers

- **Automatic**: Every push to `main` branch
- **Manual**: Workflow dispatch for on-demand runs

---

## Project Structure

```
gamble-comment-detector/
├── 📁 .github/
│   └── workflows/
│       └── pipeline.yml   # Pipeline Configuration
├── 📁 app/
│   ├── __init__.py
│   └── main.py            # FastAPI application
├── 📁 data/
│   └── comments.csv       # Training dataset
├── 📁 model/
│   ├── eval_report.json   # Automated evaluation metrics
│   ├── saved_model.joblib # Auto-generated ML model
│   └── vectorizer.joblib  # Auto-generated text vectorizer
├── 📁 notebooks/
│   └── Baseline.ipynb     # Jupyter notebook for experimentation
├── 📁 scripts/
│   └── generate_dummy_data.py  # Auto data generation
├── 📁 src/
│   ├── __init__.py
│   ├── evaluate.py        # Automated model evaluation
│   ├── inference.py       # Prediction logic
│   ├── preprocessing.py   # Data preprocessing
│   └── train.py          # Automated model training
├── app.py                # Main application entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Analyze a single comment |
| `POST` | `/predict/batch` | Analyze multiple comments |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/model/info` | Get model information and version |

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
    "model_version": "v1.2.3",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## MLOps Workflow Details

### GitHub Actions Pipeline Configuration

The automated pipeline (`pipeline.yml`) includes:

```yaml
# Key Pipeline Steps
1. Environment Setup (Python 3.9)
2. Dependency Installation
3. Automated Model Training
4. Performance Evaluation
5. Model Artifact Versioning
6. Auto-Deployment to HF Spaces
```

### Required Secrets

For the pipeline to work, set these GitHub repository secrets:

- `HF_TOKEN`: Hugging Face API token for deployment

---

## Development

### Local Development
```bash
# Run tests
pytest tests/ -v

# Code formatting
black src/ app/
flake8 src/ app/

# Generate dummy data
python scripts/generate_dummy_data.py
```

### Pipeline Testing
```bash
# Test pipeline locally (requires Act)
act -j train-and-deploy
```

---

## Contributing

We welcome contributions! The automated pipeline will test your changes automatically.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/-------`)
3. **Commit** your changes (`git commit -m '-----------'`)
4. **Push** to the branch (`git push origin feature/---------`)
5. **Open** a Pull Request

---

## Roadmap

- [x] ✅ Basic ML model implementation
- [x] ✅ FastAPI REST API
- [x] ✅ Automated MLOps pipeline with GitHub Actions
- [x] ✅ Auto-deployment to Hugging Face Spaces
- [x] ✅ Model evaluation metrics and reporting
- [ ] 🔄 A/B testing framework
- [ ] 🔄 Model performance monitoring dashboard
- [ ] 🔄 Advanced deep learning models
- [ ] 🔄 Multi-language support
- [ ] 🔄 Docker containerization
- [ ] 🔄 Kubernetes deployment

---

## Live Demo

**Try the live model**: [Hugging Face Spaces](https://huggingface.co/spaces/minggo-commits/Gamble-Comment-Detector)

*Updated automatically with every model improvement!*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Thanks to all contributors who helped build this project
- Built with using FastAPI, scikit-learn, GitHub Actions, and Hugging Face

---

## Support

- **Found a bug?** [Open an issue](https://github.com/minggo-commits/gamble-comment-detector/issues)
- **Have a feature request?** [Start a discussion](https://github.com/minggo-commits/gamble-comment-detector/discussions)
- **Pipeline Issues?** Check [Actions tab](https://github.com/minggo-commits/gamble-comment-detector/actions)

---

<p align="center">
  <strong>⭐ Don't forget to star this repository if you found it helpful!</strong><br>
  <e
