# AI-Powered Medical Imaging Diagnostic System

An end-to-end clinical workstation for Pneumonia detection from Chest X-rays. This platform integrates deep learning (ResNet50) for automated diagnosis and Explainable AI (Grad-CAM) to provide clinicians with interpretable heatmaps that highlight image regions influencing model predictions.

## 🌟 Key Features

- Automated diagnosis: Fine-tuned ResNet50 classification (NORMAL vs PNEUMONIA).
- Interpretability (Grad-CAM): Generates heatmaps to explain model decisions and support clinician review.
- Scalable data management: MongoDB + GridFS for high-resolution image storage and patient metadata.
- Dual-portal workflow: Separate Nurse/Admin and Physician interfaces for intake, upload, search, and review.
- Lightweight UI: Streamlit-based interface for quick deployment and interactive inspection.

## 🛠️ Technical Stack

- Framework: PyTorch
- Computer Vision: OpenCV, Torchvision
- Backend & Database: MongoDB, GridFS
- UI / Deployment: Streamlit
- Language: Python 3.8+

## 📐 System Overview

- Preprocessing: Image normalization and resizing using torchvision.transforms.
- Inference: ResNet50 outputs a probability for NORMAL vs PNEUMONIA.
- Explainability: Grad-CAM computed from gradients of the final convolutional layer to produce heatmaps.
- Storage: Patient metadata in MongoDB collections; raw image binaries managed via GridFS.

---

## 🚀 Quickstart

Prerequisites

- Python 3.8 or later
- MongoDB installed and running (local or remote)
- (Optional) GPU with CUDA for faster inference/training
- Model weights: `pneumonia_resnet50.pth` (place in the `models/` directory or set the path via environment/config)

Installation

1. Clone the repository:

```bash
git clone https://github.com/druthivaranasi15/AI-Powered-Medical-Imaging-Diagnostic-System.git
cd AI-Powered-Medical-Imaging-Diagnostic-System
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.\.venv\Scripts\activate   # Windows (PowerShell/CMD)
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start MongoDB (example for local Windows service):

```bash
# Windows (if MongoDB is installed as a service)
net start MongoDB

# macOS / Linux (example using brew or systemctl)
brew services start mongodb-community
# or
sudo systemctl start mongod
```

5. Run the Streamlit app:

```bash
streamlit run app.py
```

Notes

- If you do not have the pretrained model file, see the `models/` folder or training notebooks to reproduce weights. You can also configure the app to point to a remote weights URL or a model registry.

---

## 🔎 Usage

- Nurse/Admin Tab: Upload patient images (DICOM/JPG) and enter metadata.
- Physician Tab: Search patient records, run inference on images, and view Grad-CAM heatmaps for interpretability.
- The UI allows downloading heatmaps and storing both the original and visualizations back to GridFS.

## 📂 Project Structure (high level)

- `app.py` — Streamlit UI and navigation
- `processor.py` — Image preprocessing, transformations, and Grad-CAM logic
- `model.py` — Model architecture and weight-loading utilities
- `inference.py` — Prediction and post-processing logic
- `database.py` — MongoDB connection and GridFS handlers
- `notebooks/` — Jupyter notebooks for exploration, training, and evaluation
- `models/` — Model weight files (e.g. `pneumonia_resnet50.pth`)
- `requirements.txt` — Python dependencies

If file names differ in the repository, refer to the actual file list — these are the canonical modules used by the app.

## Contributing

Contributions, issues, and feature requests are welcome. Please open an issue to discuss changes before submitting pull requests. Include reproducible steps and sample data when possible (obeying any patient privacy constraints).

## License

Author: Druthi Varanasi

This project is provided for educational and research purposes. If you intend to use it in a clinical setting, validate thoroughly and obtain required regulatory approvals. If you would like a formal open-source license added, tell me which license (MIT, Apache-2.0, GPL-3.0, etc.) and I can add it.

## Contact

For questions or collaboration, contact: druthivaranasi15 (GitHub) or open an issue on this repository.
