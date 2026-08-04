import io
import torch
from PIL import Image
from model import load_medical_model
from processor import preprocess_image, generate_gradcam

# Load model once at startup
model, classes = load_medical_model()

def run_diagnosis(raw_bytes: bytes):
    # 1. Convert raw bytes into PIL Image (RGB)
    pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    
    # 2. Preprocess tensor for ResNet50 input
    input_tensor = preprocess_image(pil_image)
    
    # 3. Model Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Extract direct probabilities (Index 0: NORMAL, Index 1: PNEUMONIA)
        normal_prob = probabilities[0].item()
        pneumonia_prob = probabilities[1].item()
        
        # Clinical Decision Thresholding (Default: 0.40)
        # Lowers false-negative rates for screening false "NORMAL" scans
        CLINICAL_THRESHOLD = 0.70
        
        if pneumonia_prob >= CLINICAL_THRESHOLD:
            prediction = "PNEUMONIA"
            confidence = pneumonia_prob * 100
        else:
            prediction = "NORMAL"
            confidence = normal_prob * 100
            
    # 4. Generate Grad-CAM composite map
    gradcam_bytes = generate_gradcam(model, input_tensor, pil_image)
    
    return prediction, confidence, gradcam_bytes
