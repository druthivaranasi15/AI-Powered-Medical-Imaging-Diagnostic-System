# AI-Powered Medical Imaging Diagnostic System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11%2B-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/Status-Research-yellow.svg)](#medical-disclaimer)

An end-to-end clinical workstation for detecting Pneumonia from Chest X-rays. The project combines a fine-tuned ResNet50 classifier with Grad-CAM explainability, a lightweight Streamlit UI, and MongoDB-backed data management. Designed for radiologists and clinicians to aid diagnostic workflows with interpretable AI predictions and comprehensive audit trails.

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This project is a **research prototype** and is **NOT** approved for clinical use without proper regulatory validation.

- ❌ **Not a replacement** for professional radiologist judgment
- ❌ **Not validated** for production clinical environments
- ✅ **For research and educational purposes only**
- ✅ Requires external clinical validation before deployment
- ✅ Subject to regulatory approval (FDA, CE, etc.) before medical use

Always consult qualified medical professionals. Any deployment must comply with local healthcare regulations (HIPAA, GDPR, etc.).

---

## 📋 What This Project Does

- **Ingests** chest X-ray images (DICOM/JPG) and patient metadata via a Nurse/Admin portal
- **Stores** images and metadata in MongoDB + GridFS for retrieval, auditability, and HIPAA-compliant access logging
- **Runs inference** with a ResNet50-based classifier and returns probability scores for NORMAL vs PNEUMONIA classification
- **Produces Grad-CAM heatmaps** to highlight image regions that contributed to predictions, enabling clinician interpretability and trust
- **Maintains audit trails** for all diagnostic decisions and metadata modifications

---

## 📊 Model Performance

Evaluation metrics from the training notebook (see `notebooks/training-model.ipynb`):

### Summary Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Set Size** | 624 images | NORMAL: 234, PNEUMONIA: 390 |
| **Overall Accuracy** | 0.88 (88%) | Overall correctness across all samples |
| **ROC-AUC** | 0.964 | Excellent discrimination ability |
| **Sensitivity** (PNEUMONIA Recall) | ~0.98 (98%) | High true positive rate (screening priority) |
| **Specificity** (NORMAL Recall) | ~0.71 (71%) | Lower due to class imbalance |
| **Decision Threshold** | 0.70 | Optimized for sensitivity; tune per use-case |

### Detailed Classification Report

```
--- FINAL CLINICAL EVALUATION METRICS (TEST SET) ---
              precision    recall  f1-score   support

      NORMAL       0.96      0.71      0.82       234
   PNEUMONIA       0.85      0.98      0.91       390

    accuracy                           0.88       624
   macro avg       0.90      0.85      0.86       624
weighted avg       0.89      0.88      0.88       624
```

#### Metric Interpretation

| Class | Precision | Recall | F1-Score | Support | Interpretation |
|-------|-----------|--------|----------|---------|-----------------|
| **NORMAL** | 0.96 | 0.71 | 0.82 | 234 | 96% of predicted NORMAL cases are correct; 71% of actual NORMAL cases are detected |
| **PNEUMONIA** | 0.85 | 0.98 | 0.91 | 390 | 85% of predicted PNEUMONIA cases are correct; 98% of actual PNEUMONIA cases are detected |
| **Weighted Avg** | 0.89 | 0.88 | 0.88 | 624 | Overall performance accounting for class imbalance |

###  Performance Analysis

**Why is NORMAL specificity lower (71%)?**
- The test set is heavily imbalanced: 390 PNEUMONIA vs 234 NORMAL samples (62.5% vs 37.5%)
- The model is optimized for sensitivity (recall of PNEUMONIA) to prioritize screening—missing pneumonia cases is more critical than false positives
- Decision threshold of 0.70 prioritizes sensitivity; can be tuned to improve specificity if needed

**Key Insights:**
- ✅ **High PNEUMONIA Recall (98%)**: Excellent for screening—catches 98% of actual pneumonia cases
- ✅ **High NORMAL Precision (96%)**: When the model predicts NORMAL, it's correct 96% of the time
- ⚠️ **Lower NORMAL Recall (71%)**: Some normal cases are misclassified as PNEUMONIA (false positives)
- ✅ **Balanced F1-Scores**: Both classes have reasonable F1 scores (0.82 for NORMAL, 0.91 for PNEUMONIA)

**Recommended Use:**
- Use this model for **screening and initial triage** in clinical workflows
- A radiologist should always review AI predictions, especially for NORMAL predictions (when 29% might be missed)
- Consider the 0.70 threshold tunable per clinical context

---

## 🎯 Key Features

- ✅ **Interpretable AI**: Grad-CAM heatmaps for every prediction
- ✅ **Scalable Backend**: MongoDB + GridFS for large-scale image storage
- ✅ **User-Friendly UI**: Streamlit web interface for nurses, admins, and radiologists
- ✅ **Audit Logging**: Complete tracking of images, predictions, and metadata changes
- ✅ **DICOM Support**: Native ingestion of medical imaging standards
- ✅ **Fast Inference**: Optimized ResNet50 model with minimal latency
- ✅ **Role-Based Access**: Nurse/Admin portals with permission controls

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- TensorFlow 2.11+
- MongoDB 4.4+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/druthivaranasi15/AI-Powered-Medical-Imaging-Diagnostic-System.git
   cd AI-Powered-Medical-Imaging-Diagnostic-System
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB connection string and model paths
   ```

5. **Download the pre-trained model**
   ```bash
   python download_model.py  # Downloads ResNet50 checkpoint
   ```

### Running the Application

```bash
streamlit run src/ui/app.py
```

The application will launch at `http://localhost:8501`

### Using the Demo

1. Navigate to the Nurse Portal
2. Upload a chest X-ray image (DICOM or JPG)
3. Enter patient metadata (age, gender, etc.)
4. Click "Analyze"
5. View the prediction probability and Grad-CAM heatmap
6. Admin Portal allows review and audit log access

---

## 📁 Project Structure

```
AI-Powered-Medical-Imaging-Diagnostic-System/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment configuration template
├── notebooks/
│   ├── training-model.ipynb          # Model training & evaluation
│   ├── data-exploration.ipynb        # Dataset analysis
│   └── results-visualization.ipynb   # Performance plots & heatmaps
├── models/
│   ├── resnet50_pneumonia.h5         # Trained model weights
│   ├── preprocessing_config.json     # CLAHE & normalization params
│   └── class_weights.pkl             # Class balancing weights
├── src/
│   ├── inference/
│   │   ├── model_loader.py           # Load trained model
│   │   ├── predictor.py              # Inference pipeline
│   │   └── gradcam.py                # Grad-CAM heatmap generation
│   ├── data/
│   │   ├── dicom_loader.py           # DICOM/JPG ingestion
│   │   ├── preprocessing.py          # Image normalization & CLAHE
│   │   └── mongodb_handler.py        # GridFS storage & retrieval
│   ├── auth/
│   │   ├── access_control.py         # Role-based permissions
│   │   └── audit_logger.py           # Compliance logging
│   └── utils/
│       └── config.py                 # Centralized configuration
├── ui/
│   ├── app.py                        # Main Streamlit app
│   ├── nurse_portal.py               # Nurse interface
│   ├── admin_portal.py               # Admin dashboard
│   └── components/
│       ├── image_uploader.py
│       ├── prediction_display.py
│       └── audit_log_viewer.py
├── tests/
│   ├── test_inference.py             # Unit tests for predictor
│   ├── test_preprocessing.py         # Data pipeline tests
│   └── test_api.py                   # Integration tests
├── docker/
│   ├── Dockerfile                    # Containerization
│   └── docker-compose.yml            # Multi-container setup (app + MongoDB)
├── docs/
│   ├── DEPLOYMENT.md                 # Production deployment guide
│   ├── API.md                        # API documentation
│   └── CONTRIBUTING.md               # Development guidelines
└── LICENSE                           # MIT License
```

---

## 🏗️ Architecture & Model Details

### Model Architecture
- **Backbone**: ResNet50 (pre-trained on ImageNet, fine-tuned on chest X-rays)
- **Input Shape**: 224 × 224 × 3 (RGB)
- **Output**: Binary classification (NORMAL vs PNEUMONIA) with probability scores
- **Preprocessing**:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement
  - Normalization using ImageNet mean/std
  - Resize to 224×224 with aspect ratio preservation

### Data Pipeline
```
Raw DICOM/JPG → DICOM Parser → CLAHE Enhancement → Resizing → 
Model Inference → Prediction Score → Grad-CAM Generation → 
MongoDB Storage + Audit Log → UI Display
```

### Explainability
- **Grad-CAM (Gradient-weighted Class Activation Mapping)** highlights which regions of the X-ray influenced the prediction
- Helps radiologists understand model decisions and identify potential failure modes
- Increases clinician trust in AI-assisted diagnostics

---

## 🔐 Security & Compliance

### Data Privacy
- ✅ Patient metadata encrypted at rest (MongoDB)
- ✅ HIPAA-compliant audit logging for all image access
- ✅ GDPR-compliant data retention policies
- ✅ Role-based access control (RBAC) for nurses/admins/radiologists

### Access Control
- Nurse Portal: Limited to image upload and metadata entry
- Admin Portal: Full audit log access, user management
- Radiologist Portal: Review predictions, override decisions, export reports

### Audit Trail
Every action logged with:
- Timestamp, user ID, action type
- Before/after values for metadata changes
- Image access history and retention periods

---

## 📈 Known Limitations & Future Improvements

### Current Limitations
1. **Class Imbalance**: Lower specificity (~71%) on NORMAL cases due to dataset imbalance
2. **Single-Task Model**: Currently detects only NORMAL vs PNEUMONIA; does not classify other chest pathologies
3. **Image Quality Sensitivity**: Performance degrades on low-quality, rotated, or non-standard X-rays
4. **Equipment Variation**: Model trained on specific equipment; may not generalize to all hospitals
5. **Not Clinically Validated**: Research prototype; requires external validation before clinical use

### Future Roadmap
- [ ] Multi-class classification (Pneumonia, TB, COVID-19, etc.)
- [ ] Federated learning for privacy-preserving multi-site training
- [ ] Uncertainty quantification (Bayesian deep learning)
- [ ] Model interpretability enhancements (SHAP, attention maps)
- [ ] Mobile app for point-of-care screening
- [ ] Real-time performance monitoring dashboard
- [ ] Integration with DICOM viewers (OHIF, Weasis)

---

## 📚 Usage & Examples

### Python API (Programmatic Usage)

```python
from src.inference.predictor import Predictor
from src.data.dicom_loader import load_dicom

# Initialize predictor
predictor = Predictor(model_path='models/resnet50_pneumonia.h5')

# Load and preprocess image
image_array = load_dicom('path/to/chest_xray.dcm')

# Run inference
prediction = predictor.predict(image_array)
print(f"Class: {prediction['class']}, Confidence: {prediction['probability']:.2%}")

# Generate Grad-CAM heatmap
heatmap = predictor.generate_gradcam(image_array)
```

### Web Interface
See `src/ui/app.py` for Streamlit interface usage and screenshots.

---

## 🧪 Testing

Run the test suite to validate model and pipeline:

```bash
pytest tests/ -v
pytest tests/test_inference.py -v         # Model inference tests
pytest tests/test_preprocessing.py -v    # Data pipeline tests
pytest tests/test_api.py -v              # Integration tests
```

---

## 📋 Requirements

See `requirements.txt` for the full dependency list:
- TensorFlow 2.11+
- scikit-learn (preprocessing, metrics)
- opencv-python (image processing)
- pymongo (database)
- streamlit (UI)
- numpy, pandas, matplotlib (utilities)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up the development environment
- Code style and testing standards
- Submitting pull requests
- Reporting issues

---

## 📖 Additional Documentation

- [**DEPLOYMENT.md**](docs/DEPLOYMENT.md) — Production deployment, Docker setup, scalability
- [**API.md**](docs/API.md) — REST API endpoints and request/response formats
- [**CONTRIBUTING.md**](CONTRIBUTING.md) — Development guidelines and code conventions

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 👤 Contact & Support

For questions, issues, or collaboration opportunities:
- **GitHub Issues**: [Open an issue](https://github.com/druthivaranasi15/AI-Powered-Medical-Imaging-Diagnostic-System/issues)
- **Email**: [Your Email]
- **LinkedIn**: [Your LinkedIn Profile]

---

## 🙏 Acknowledgments

- **Dataset**: ChexPert, NIH Chest X-ray Dataset
- **Framework**: TensorFlow & Keras team
- **UI**: Streamlit community
- **Medical Imaging**: Thanks to radiologists who provided domain expertise

---

**Last Updated**: August 2026
**Status**: Research Prototype — Not for Clinical Use
