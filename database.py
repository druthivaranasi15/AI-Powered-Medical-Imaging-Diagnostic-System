import pymongo
import gridfs

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Medical_AI_System"]
fs = gridfs.GridFS(db)

def save_patient_data(name, age, image_path):
    # 1. Save Image to GridFS
    with open(image_path, 'rb') as f:
        image_id = fs.put(f, filename=f"{name}_xray.jpg")
    
    # 2. Save Metadata linked to Image ID
    patient_record = {
        "name": name,
        "age": age,
        "image_id": image_id
    }
    db.patients.insert_one(patient_record)
    print(f"✅ Record created for {name}. Image ID: {image_id}")
    return image_id