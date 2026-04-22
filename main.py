from database import save_patient_data
from inference import run_diagnosis

if __name__ == "__main__":
    print("🚀 Initializing Medical System...")

    # Step 1: Save patient + image
    image_id = save_patient_data(
        name="John Doe",
        age=45,
        image_path="test_xray.jpg"
    )

    # Step 2: Run diagnosis
    result = run_diagnosis("John Doe")

    if result:
        print(f"\n🎯 Final Result: {result}")
    else:
        print("\n❌ Diagnosis failed — check errors above")