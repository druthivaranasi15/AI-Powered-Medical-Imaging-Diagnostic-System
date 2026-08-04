from database import save_patient_data
from inference import run_diagnosis

if __name__ == "__main__":
    print("🚀 Initializing Medical System...")

    patient_id, image_id = save_patient_data(
        name="John Doe",
        age=45,
        image_path="test_xray.jpg"
    )

    result = run_diagnosis(patient_id)

    if result:
        print(f"\n🎯 Final Result: {result['diagnosis_text']}")
    else:
        print("\n❌ Diagnosis failed — check errors above")
