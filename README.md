# AI-Powered Medical Imaging Diagnostic System

An end-to-end clinical workstation for detecting Pneumonia from Chest X‑rays. The project combines a fine-tuned ResNet50 classifier with Grad‑CAM explainability, a lightweight Streamlit UI, and MongoDB/GridFS for scalable storage. It is intended as a research/educational platform to support clinician review and triage workflows (not a certified clinical device).

## What this project does (short)

- Ingests chest X‑ray images (DICOM/JPG) and patient metadata via a Nurse/Admin portal.
- Stores images and metadata in MongoDB + GridFS for retrieval and auditability.
- Runs inference with a ResNet50-based classifier and returns a probability for NORMAL vs PNEUMONIA.
- Produces Grad‑CAM heatmaps to highlight image regions that contributed to predictions, aiding interpretability and clinician trust.

---

## ⚙️ Results (concise)

These evaluation numbers were produced by the training notebook (see `notebooks/` and `training-model.ipynb`):

- Test set: 624 images (NORMAL: 234, PNEUMONIA: 390)
- Accuracy: 0.88
- ROC-AUC: 0.964
- Sensitivity (PNEUMONIA recall): ~0.98
- Specificity (NORMAL recall): ~0.71
- Deployed decision threshold: 0.70 (prioritizes sensitivity for screening; tune per use-case)

Notes: metrics depend on the dataset, preprocessing (CLAHE, resizing), and training hyperparameters. Perform external validation before any real-world or clinical use.

---

(Other sections — Features, Quickstart, Usage, Project structure, Contributing, License, Contact — remain unchanged.)
