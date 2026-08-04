import gridfs
from bson.objectid import ObjectId
from pymongo import MongoClient

# Initialize MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["medical_db"]
fs = gridfs.GridFS(db)
scans_collection = db["patient_scans"]

def save_scan(patient_name, age, raw_img_bytes, gradcam_bytes, prediction, confidence):
    """Saves scan and image metadata, returning a unique record ID."""
    raw_id = fs.put(raw_img_bytes, filename=f"{patient_name}_raw.png")
    gradcam_id = fs.put(gradcam_bytes, filename=f"{patient_name}_gradcam.png")
    
    record = {
        "patient_name": patient_name,
        "age": age,
        "prediction": prediction,
        "confidence": confidence,
        "raw_img_id": raw_id,
        "gradcam_img_id": gradcam_id
    }
    
    result = scans_collection.insert_one(record)
    return str(result.inserted_id)  # Return unique MongoDB ObjectId string

def get_all_patients():
    """Returns list of patients formatted with unique IDs for dropdown menus."""
    records = list(scans_collection.find({}, {"_id": 1, "patient_name": 1, "age": 1}))
    # Example format: "John Doe (Age: 45) - ID: 651a2b..."
    return [
        {
            "id": str(r["_id"]),
            "display_label": f"{r['patient_name']} (Age: {r.get('age', 'N/A')}) — ID: {str(r['_id'])[-6:]}"
        }
        for r in records
    ]

def get_scan_by_id(record_id):
    """Retrieves patient record using the unique ObjectId."""
    return scans_collection.find_one({"_id": ObjectId(record_id)})
