import torch
from database import db, fs
from processor import preprocess_image
from model import load_medical_model

try:
    _model, _classes = load_medical_model()
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise

def run_diagnosis(patient_name: str):
    print(f"\n🔍 Analyzing: {patient_name}...")
    try:
        patient = db.patients.find_one({"name": patient_name})
        if not patient:
            print(f"❌ Patient '{patient_name}' not found.")
            return None

        image_bytes  = fs.get(patient['image_id']).read()
        image_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            outputs    = _model(image_tensor)
            probs      = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx   = probs.argmax().item()
            prediction = _classes[pred_idx]
            confidence = probs[pred_idx].item() * 100

        diagnosis_text = f"{prediction} ({confidence:.2f}% confidence)"

        db.patients.update_one(
            {"_id": patient["_id"]},
            {"$set": {
                "prediction":       diagnosis_text,
                "normal_prob":      round(probs[0].item() * 100, 2),
                "pneumonia_prob":   round(probs[1].item() * 100, 2),
            }}
        )
        print(f"🏥 {diagnosis_text}")
        return diagnosis_text

    except Exception as e:
        import traceback
        print(f"❌ Diagnosis error: {e}")
        traceback.print_exc()
        return None