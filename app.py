from ultralytics import YOLO
from PIL import Image
from collections import defaultdict
import streamlit as st
import time

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Rice Disease Detection",
    page_icon="🌾",
    layout="centered"
)

# ---------------------------------
# TITLE
# ---------------------------------
st.title("🌾 Rice Disease Detection System")

st.write(
    "Upload a rice leaf image to detect rice diseases using YOLOv8."
)

# ---------------------------------
# CUSTOM CLASS NAMES
# ---------------------------------
CORRECT_NAMES = {
    0: 'dirty_panicle',
    1: 'rice_blast',
    2: 'narrow_brown',
    3: 'brown_spot'
}

# ---------------------------------
# DISEASE INFORMATION
# ---------------------------------
DISEASE_INFO = {

    "dirty_panicle": {
        "description":
        "Dirty panicle is a fungal disease affecting rice grains and panicles.",
        
        "treatment":
        "Use clean seeds, proper field sanitation, and fungicides."
    },

    "rice_blast": {
        "description":
        "Rice blast is a serious fungal disease causing lesions on leaves.",
        
        "treatment":
        "Apply Tricyclazole fungicide and avoid excessive nitrogen fertilizer."
    },

    "narrow_brown": {
        "description":
        "Narrow brown spot causes elongated brown lesions on rice leaves.",
        
        "treatment":
        "Maintain balanced fertilization and apply recommended fungicides."
    },

    "brown_spot": {
        "description":
        "Brown spot causes circular brown lesions and weakens rice plants.",
        
        "treatment":
        "Improve soil nutrients and apply fungicide if necessary."
    }
}

# ---------------------------------
# LOAD MODEL
# ---------------------------------
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.header("⚙️ Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)

# ---------------------------------
# FILE UPLOADER
# ---------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Rice Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# MAIN PROCESS
# ---------------------------------
if uploaded_file is not None:

    # ---------------------------------
    # LOAD IMAGE
    # ---------------------------------
    image = Image.open(uploaded_file).convert("RGB")

    # FAST RESIZE
    image.thumbnail((640, 640))

    # ---------------------------------
    # SHOW IMAGE
    # ---------------------------------
    st.image(
        image,
        caption="📷 Uploaded Image",
        use_container_width=True
    )

    # ---------------------------------
    # START TIMER
    # ---------------------------------
    start_time = time.time()

    # ---------------------------------
    # PREDICT
    # ---------------------------------
    with st.spinner("🔍 Detecting diseases..."):

        results = model.predict(
            source=image,
            imgsz=320,
            conf=confidence_threshold,
            verbose=False
        )

        result = results[0]

    # ---------------------------------
    # END TIMER
    # ---------------------------------
    end_time = time.time()

    detection_time = end_time - start_time

    # ---------------------------------
    # SHOW RESULT IMAGE
    # ---------------------------------
    plotted_image = result.plot()

    st.image(
        plotted_image,
        caption="🧠 Detection Result",
        use_container_width=True
    )

    # ---------------------------------
    # DETECTION SPEED
    # ---------------------------------
    st.success(
        f"⚡ Detection completed in "
        f"{detection_time:.2f} seconds"
    )

    # ---------------------------------
    # DETECTION SUMMARY
    # ---------------------------------
    st.subheader("🦠 Detection Summary")

    if result.boxes is not None and len(result.boxes) > 0:

        disease_summary = defaultdict(list)

        # ---------------------------------
        # PROCESS DETECTIONS
        # ---------------------------------
        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = CORRECT_NAMES.get(
                class_id,
                f"Unknown_{class_id}"
            )

            disease_summary[class_name].append(confidence)

        # ---------------------------------
        # SHOW SUMMARIZED RESULTS
        # ---------------------------------
        total_detections = 0

        for disease, confidences in disease_summary.items():

            highest_conf = max(confidences)

            total_spots = len(confidences)

            total_detections += total_spots

            st.success(
                f"✅ {disease.upper()} detected "
                f"({total_spots} spot/s found)\n"
                f"Highest Confidence: {highest_conf:.2f}"
            )

            # ---------------------------------
            # DISEASE DESCRIPTION
            # ---------------------------------
            st.info(
                f"📖 Description:\n"
                f"{DISEASE_INFO[disease]['description']}"
            )

            # ---------------------------------
            # TREATMENT RECOMMENDATION
            # ---------------------------------
            st.warning(
                f"💊 Treatment Recommendation:\n"
                f"{DISEASE_INFO[disease]['treatment']}"
            )

        # ---------------------------------
        # SEVERITY ANALYSIS
        # ---------------------------------
        st.subheader("📊 Severity Analysis")

        if total_detections <= 2:
            severity = "🟢 Low Infection"

        elif total_detections <= 5:
            severity = "🟡 Moderate Infection"

        else:
            severity = "🔴 Severe Infection"

        st.info(
            f"Total Detected Spots: {total_detections}\n\n"
            f"Estimated Severity: {severity}"
        )

        # ---------------------------------
        # DETAILED DETECTIONS
        # ---------------------------------
        with st.expander("📋 Show Detailed Detections"):

            detection_number = 1

            for disease, confidences in disease_summary.items():

                st.write(f"### 🌾 {disease.upper()}")

                for conf in confidences:

                    st.write(
                        f"Detection {detection_number} "
                        f"| Confidence: {conf:.2f}"
                    )

                    detection_number += 1

    else:
        st.warning("⚠️ No disease detected.")

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")

st.caption(
    "Rice Disease Detection System using YOLOv8 and Streamlit"
)