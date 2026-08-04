import streamlit as st
import io
from PIL import Image
from database import save_scan, get_all_patients, get_scan_by_id, fs
from inference import run_diagnosis

st.set_page_config(page_title="Medical Diagnostic Portal", layout="wide")

# Add Navigation Tabs
tab1, tab2 = st.tabs(["Upload & Diagnose Scan", "Physician Portal"])

# --- TAB 1: DIAGNOSTIC PIPELINE ---
with tab1:
    st.title("Patient Scan Diagnostics")
    
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Full Name")
        patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
    
    uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file and patient_name:
        if st.button("Run Diagnostic Pipeline"):
            with st.spinner("Analyzing image and generating Grad-CAM..."):
                raw_bytes = uploaded_file.getvalue()
                
                # Run PyTorch Inference + GradCAM
                prediction, confidence, gradcam_bytes = run_diagnosis(raw_bytes)
                
                # Save to MongoDB & GridFS
                record_id = save_scan(patient_name, patient_age, raw_bytes, gradcam_bytes, prediction, confidence)
                
                st.success(f"Scan processed and saved successfully! Record ID: `{record_id}`")
                
                # Display Results
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.image(raw_bytes, caption="Uploaded X-Ray", use_container_width=True)
                with res_col2:
                    st.image(gradcam_bytes, caption="Grad-CAM Heatmap", use_container_width=True)

# --- TAB 2: PHYSICIAN PORTAL ---
with tab2:
    st.title("Physician Portal")
    
    patients = get_all_patients()
    
    if not patients:
        st.info("No patient records found in the database. Use the 'Upload & Diagnose Scan' tab to add patient records.")
    else:
        patient_map = {p["display_label"]: p["id"] for p in patients}
        selected_label = st.selectbox("Select Patient Record:", list(patient_map.keys()))
        selected_id = patient_map[selected_label]
        
        record = get_scan_by_id(selected_id)
        
        if record:
            st.subheader(f"Patient Name: {record['patient_name']}")
            st.write(f"**Age:** {record.get('age', 'N/A')}")
            st.write(f"**Diagnosis:** {record['prediction']} ({record['confidence']:.2f}% confidence)")
            st.write(f"**System Record ID:** `{record['_id']}`")
            
            raw_data = fs.get(record['raw_img_id']).read()
            gradcam_data = fs.get(record['gradcam_img_id']).read()
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.image(raw_data, caption="Original X-Ray", use_container_width=True)
            with p_col2:
                st.image(gradcam_data, caption="Grad-CAM Visualization", use_container_width=True)
