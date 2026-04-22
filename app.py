import streamlit as st
import os
import io
import cv2
import numpy as np
from PIL import Image
from database import save_patient_data, db, fs
from inference import run_diagnosis
from model import load_medical_model
from processor import preprocess_image, generate_heatmap

st.set_page_config(
    page_title="Medical AI System",
    layout="wide",
    page_icon="🏥"
)

st.title("🏥 Intelligent Medical Imaging Platform")

tab_upload, tab_doctor = st.tabs(["📤 New Patient Analysis", "🔍 Physician's Portal"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PATIENT / NURSE (Clean, simple, no heatmap)
# ══════════════════════════════════════════════════════════════════
with tab_upload:
    st.header("🩺 New Patient X-Ray Submission")

    col_form, col_preview = st.columns([1, 1])

    with col_form:
        with st.form("patient_form", clear_on_submit=False):
            name   = st.text_input("Patient Full Name")
            age    = st.number_input("Patient Age", min_value=0, max_value=120, step=1)
            file   = st.file_uploader("Select Chest X-Ray", type=["jpg", "png", "jpeg"])
            submit = st.form_submit_button("🚀 Submit & Run AI Diagnosis")

    with col_preview:
        st.subheader("📷 X-Ray Preview")
        if file:
            st.image(file, caption="Uploaded X-Ray", use_container_width=True)
        else:
            st.info("📷 Uploaded image will appear here once selected.")

    if submit:
        if not name:
            st.error("⚠️ Please enter the patient name.")
        elif not file:
            st.error("⚠️ Please upload a chest X-ray image.")
        else:
            temp_path = "temp_upload.jpg"
            with open(temp_path, "wb") as tmp:
                tmp.write(file.getbuffer())

            with st.spinner("💾 Saving record to database..."):
                save_patient_data(name, int(age), temp_path)

            with st.spinner("🤖 Running AI diagnosis — please wait..."):
                result = run_diagnosis(name)

            os.remove(temp_path)

            if result:
                saved     = db.patients.find_one({"name": name})
                n_prob    = saved.get("normal_prob",    0.0)
                p_prob    = saved.get("pneumonia_prob", 0.0)
                is_pneumo = "PNEUMONIA" in result.upper()

                st.markdown("---")

                # ── Status banner ─────────────────────────────────
                if is_pneumo:
                    st.error("🔴 PNEUMONIA DETECTED — Please consult a doctor immediately.")
                else:
                    st.success("🟢 NORMAL — No signs of pneumonia detected.")

                # ── Metrics ───────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("👤 Patient",   name)
                m2.metric("🎂 Age",       int(age))
                m3.metric("🏥 AI Result", result.split("(")[0].strip())

                # ── Confidence bars ───────────────────────────────
                st.markdown("#### 📊 Confidence Breakdown")
                b1, b2 = st.columns(2)
                with b1:
                    st.markdown("🟢 **NORMAL**")
                    st.progress(int(n_prob), text=f"{n_prob}%")
                with b2:
                    st.markdown("🔴 **PNEUMONIA**")
                    st.progress(int(p_prob), text=f"{p_prob}%")

                # ── Report card ───────────────────────────────────
                st.markdown("---")
                st.markdown("### 📋 Your AI Diagnostic Report")
                with st.container(border=True):
                    r1, r2 = st.columns(2)
                    with r1:
                        st.markdown(f"**Patient Name:** {name}")
                        st.markdown(f"**Patient Age:** {int(age)} years")
                        st.markdown(f"**AI Diagnosis:** `{result.split('(')[0].strip()}`")
                        st.markdown(f"**Confidence:** `{result.split('(')[1].replace(')', '')}`")
                    with r2:
                        st.markdown(f"**Normal Probability:** {n_prob}%")
                        st.markdown(f"**Pneumonia Probability:** {p_prob}%")
                        st.markdown(f"**Record Saved:** ✅ Yes")
                        st.markdown(f"**Database:** Medical_AI_System")

                # ── Patient advisory ──────────────────────────────
                st.markdown("---")
                st.markdown("### 💬 What This Means For You")
                if is_pneumo:
                    st.warning("""
                    **⚠️ Action Required:**
                    - The AI has detected signs of **pneumonia** in your X-ray.
                    - Please **do not ignore** this result — schedule a doctor visit immediately.
                    - Bring this report to your physician for confirmation and treatment.
                    - Stay hydrated, rest, and avoid strenuous activity until reviewed.
                    - Your record has been saved and your doctor can access it via the Physician's Portal.
                    """)
                else:
                    st.info("""
                    **✅ Good News:**
                    - Your X-ray appears **normal** with no signs of pneumonia detected.
                    - Continue maintaining a healthy lifestyle.
                    - If you still experience symptoms, consult your doctor for further evaluation.
                    - Your record has been saved for future reference.
                    """)
            else:
                st.error("❌ Diagnosis failed. Please try again or check terminal.")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — PHYSICIAN'S PORTAL (Full details + heatmap)
# ══════════════════════════════════════════════════════════════════
with tab_doctor:
    st.header("🔍 Search & Review Patient Records")

    search_name = st.text_input(
        "Search by Patient Name",
        placeholder="Type patient name..."
    )

    if search_name:
        patient = db.patients.find_one(
            {"name": {"$regex": search_name, "$options": "i"}}
        )

        if patient:
            prediction = patient.get('prediction', 'Pending')
            is_pneumo  = "PNEUMONIA" in str(prediction).upper()
            n_prob     = patient.get('normal_prob',    0.0)
            p_prob     = patient.get('pneumonia_prob', 0.0)

            # ── Patient info bar ──────────────────────────────────
            st.success(f"✅ Record found for **{patient['name']}**")
            i1, i2, i3 = st.columns(3)
            i1.metric("👤 Name", patient['name'])
            i2.metric("🎂 Age",  patient.get('age', 'N/A'))
            i3.metric("🆔 ID",   str(patient['_id'])[:16] + "...")

            st.divider()

            # ── X-Ray + Quick diagnosis ───────────────────────────
            img_col, report_col = st.columns([1, 1])

            image_bytes = fs.get(patient['image_id']).read()
            raw_img     = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            with img_col:
                st.subheader("📷 Patient X-Ray")
                st.image(raw_img, use_container_width=True)

            with report_col:
                st.subheader("⚡ Quick Diagnosis")

                if prediction == 'Pending':
                    st.warning("⏳ No diagnosis yet — run from New Patient tab first.")
                else:
                    if is_pneumo:
                        st.error("🔴 PNEUMONIA DETECTED")
                    else:
                        st.success("🟢 NORMAL — No Pneumonia")

                    st.metric("Full AI Result", prediction)

                    c1, c2 = st.columns(2)
                    c1.metric("🟢 Normal",    f"{n_prob}%")
                    c2.metric("🔴 Pneumonia", f"{p_prob}%")

                    if isinstance(n_prob, float):
                        st.progress(int(n_prob), text=f"Normal: {n_prob}%")
                        st.progress(int(p_prob), text=f"Pneumonia: {p_prob}%")

                # ── Re-run button ─────────────────────────────────
                if st.button("🔄 Re-run Diagnosis", key="rerun"):
                    with st.spinner("Re-analyzing..."):
                        new_result = run_diagnosis(patient['name'])
                    if new_result:
                        st.success(f"Updated: {new_result}")
                        st.rerun()

            st.divider()

            # ── Full diagnostic report (same as user view) ────────
            st.markdown("### 📋 Full Diagnostic Report")
            with st.container(border=True):
                rd1, rd2 = st.columns(2)
                with rd1:
                    st.markdown(f"**Patient Name:** {patient['name']}")
                    st.markdown(f"**Patient Age:** {patient.get('age', 'N/A')} years")
                    st.markdown(f"**AI Diagnosis:** `{str(prediction).split('(')[0].strip()}`")
                    conf = str(prediction).split('(')[1].replace(')', '') \
                        if '(' in str(prediction) else 'N/A'
                    st.markdown(f"**Confidence:** `{conf}`")
                with rd2:
                    st.markdown(f"**Normal Probability:** {n_prob}%")
                    st.markdown(f"**Pneumonia Probability:** {p_prob}%")
                    st.markdown(f"**Record ID:** `{patient['_id']}`")
                    st.markdown(f"**Database:** Medical_AI_System")

            # ── Clinical advisory for doctor ──────────────────────
            st.markdown("### 🩺 Clinical Advisory")
            if prediction == 'Pending':
                st.info("No diagnosis available yet.")
            elif is_pneumo:
                st.error("""
                **⚠️ Clinical Action Required:**
                - AI has flagged this case as **Pneumonia** with high confidence.
                - Recommend prescribing antibiotics if bacterial pneumonia is confirmed.
                - Consider additional tests: CBC, sputum culture, CT scan if needed.
                - Monitor oxygen saturation and respiratory rate closely.
                - Schedule immediate follow-up within 24–48 hours.
                """)
            else:
                st.info("""
                **✅ Clinical Note:**
                - AI indicates **no signs of pneumonia** in this X-ray.
                - If patient presents with symptoms, consider differential diagnoses.
                - Recommend clinical correlation with physical examination findings.
                - Schedule routine follow-up if symptoms persist.
                """)

            st.divider()

            # ── Grad-CAM heatmap (doctor only) ────────────────────
            st.subheader("🔬 Grad-CAM Infection Localization")
            st.caption(
                "Highlights the exact lung regions the AI focused on "
                "to make its prediction — useful for clinical review."
            )

            if st.button("🌡️ Generate Grad-CAM Heatmap", key="doc_heatmap"):
                with st.spinner("🔬 Generating Grad-CAM analysis..."):
                    try:
                        model, _classes = load_medical_model()
                        image_tensor    = preprocess_image(image_bytes)
                        image_tensor.requires_grad_(True)

                        heatmap, pred_idx = generate_heatmap(
                            model, image_tensor, model.layer4
                        )

                        img_np          = np.array(raw_img.resize((224, 224)))
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                        heatmap_color   = cv2.applyColorMap(
                            np.uint8(255 * heatmap_resized),
                            cv2.COLORMAP_JET
                        )
                        heatmap_color   = cv2.cvtColor(
                            heatmap_color, cv2.COLOR_BGR2RGB
                        )
                        superimposed    = cv2.addWeighted(
                            img_np, 0.6, heatmap_color, 0.4, 0
                        )

                        h1, h2, h3 = st.columns(3)
                        with h1:
                            st.image(raw_img.resize((224, 224)),
                                     caption="📷 Original X-Ray",
                                     use_container_width=True)
                        with h2:
                            st.image(
                                cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR),
                                caption="🌡️ Raw Heatmap",
                                use_container_width=True,
                                channels="BGR"
                            )
                        with h3:
                            st.image(superimposed,
                                     caption="🔬 Overlay (Grad-CAM)",
                                     use_container_width=True)

                        st.warning(
                            f"🔴 **Red/warm zones** = regions AI flagged as "
                            f"**{_classes[pred_idx]}** indicators. "
                            f"🔵 **Blue/cool zones** = less significant areas."
                        )
                        st.caption(
                            "⚠️ Grad-CAM is an explainability tool to assist clinical "
                            "decision-making — not a substitute for professional diagnosis."
                        )

                    except Exception as e:
                        st.error(f"❌ Heatmap error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.error(f"❌ No record found matching **'{search_name}'**")