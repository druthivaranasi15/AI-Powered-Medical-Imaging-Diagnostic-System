# AI-Powered-Medical-Imaging-Diagnostic-System
Intelligent Medical Imaging & Diagnostic Platform
An end-to-end clinical workstation for Pneumonia detection from Chest X-rays. This platform integrates Deep Learning for automated diagnosis and Explainable AI (XAI) to provide clinicians with visual evidence for AI-generated results.

#🌟 Key Features
Automated Diagnosis: High-precision classification using a fine-tuned ResNet50 architecture.

Interpretability (Grad-CAM): Generates heatmaps highlighting symptomatic lung regions, ensuring the model is making decisions based on relevant medical features rather than noise.

Scalable Data Management: Utilizes MongoDB GridFS to handle high-resolution DICOM/JPG imagery and patient metadata.

Dual-Portal Workflow: * Nurse/Admin Tab: Streamlined patient intake and image upload.

Physician Tab: Searchable patient database with deep-dive diagnostic tools.

#🛠️ Technical Stack
Framework: PyTorch (Deep Learning)

Computer Vision: OpenCV, Torchvision

Backend & Database: MongoDB, GridFS

UI/Deployment: Streamlit

Language: Python 3.x

#📐 System Architecture
The system follows a modular architecture:

Preprocessing: Normalization and resizing using torchvision.transforms.

Inference: ResNet50 processes the image to predict NORMAL vs PNEUMONIA.

Visualization: A backward pass through the final convolutional layer calculates gradients to generate the Grad-CAM heatmap.

Storage: Metadata is stored in MongoDB collections, while the raw binary image data is managed via GridFS.

#🚀 Getting Started
Prerequisites
Python 3.8+

MongoDB installed and running locally

ResNet50 model weights (pneumonia_resnet50.pth)

** Installation **
Clone the repository:

Bash
git clone https://github.com/druthivaranasi15/Medical_AI_System.git
cd Medical_AI_System
Install dependencies:

Bash
pip install -r requirements.txt
Start MongoDB:

Bash
net start MongoDB
Run the Application:

```Bash
streamlit run app.py
```
#📂 Project Structure
app.py: Main Streamlit UI with tabbed navigation.

processor.py: Image preprocessing and Grad-CAM logic.

database.py: MongoDB connection and GridFS storage handlers.

model.py: Model architecture and weight loading.

inference.py: Logic for running predictions on new data.

📝 License
varanasi druthi
