
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Plant Health Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ---------- Hero header ---------- */
.hero {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 45%, #689f38 100%);
    padding: 2.6rem 2rem;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 28px rgba(27, 94, 32, 0.28);
}
.hero h1 {
    color: #ffffff !important;
    font-size: 2.7rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
}
.hero p {
    color: #e8f5e9 !important;
    font-size: 1.12rem;
    margin: 0;
    font-weight: 400;
}

/* ---------- Generic cards ---------- */
.card {
    background-color: rgba(127, 127, 127, 0.07);
    border: 1px solid rgba(127, 127, 127, 0.20);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
}
.card h4 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-weight: 700;
}
.card p, .card li {
    margin: 0;
    line-height: 1.55;
    opacity: 0.92;
}
.card ul {
    margin: 0.3rem 0 0 0;
    padding-left: 1.2rem;
}

/* ---------- Upload / analysis panels ---------- */
.panel-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.panel-subtext {
    opacity: 0.75;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

/* ---------- Result banner ---------- */
.result-banner {
    border-radius: 16px;
    border-left: 7px solid;
    padding: 1.5rem 1.7rem;
    margin: 0.6rem 0 1.3rem 0;
}
.result-banner h3 {
    margin: 0 0 0.25rem 0;
    font-weight: 800;
}
.result-banner span.sub {
    font-size: 0.95rem;
    opacity: 0.9;
}

/* ---------- Badges ---------- */
.badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    margin-right: 0.4rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] .stMetric {
    background-color: rgba(127, 127, 127, 0.08);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
}

/* ---------- Buttons ---------- */
.stButton>button {
    background: linear-gradient(135deg, #2e7d32 0%, #689f38 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(46, 125, 50, 0.35);
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    opacity: 0.75;
    font-size: 0.9rem;
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(127, 127, 127, 0.25);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌿 Plant Health Prediction</h1>
        <p>AI-Powered Plant Leaf Health Analysis Using Deep Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🌱 About the Project")

    st.write(
        "This system uses a Convolutional Neural Network (CNN) "
        "to analyze plant leaf images and classify their health condition."
    )

    st.divider()

    st.subheader("🔬 Detection Classes")
    st.markdown(
        "- 🟢 **Healthy**\n"
        "- ⚪ **Powdery**\n"
        "- 🟠 **Rust**"
    )

    st.divider()

    st.subheader("📊 Model Performance")
    st.metric(label="Test Accuracy", value="94%")
    st.caption("Evaluated on 150 test images.")

    st.divider()

    st.subheader("⚙️ Technology")
    st.write("TensorFlow / Keras · CNN · Streamlit")

    st.divider()

    with st.expander("🧠 How does this AI model work?"):
        st.markdown(
            "**Convolutional Neural Networks (CNNs)** are deep learning "
            "models designed to recognize visual patterns in images.\n\n"
            "**1. Convolution layers** slide small filters across the leaf "
            "image to detect low-level features such as edges, color "
            "blotches, and textures.\n\n"
            "**2. Pooling layers** shrink the feature maps, keeping the "
            "strongest signals while discarding noise, which helps the "
            "network generalize to new images.\n\n"
            "**3. Fully connected layers** combine everything the network "
            "has learned into a final decision across the three classes: "
            "*Healthy*, *Powdery*, and *Rust*.\n\n"
            "**4. Softmax output** converts raw scores into probabilities "
            "that sum to 100%, so the highest value becomes the predicted "
            "class and its size becomes the confidence."
        )

    st.divider()

    st.caption("Plant Health Prediction System — Summer Internship Project")


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_trained_model():

    model_path = "plant_disease_model.keras"

    return tf.keras.models.load_model(
        model_path
    )


try:
    model = load_trained_model()
    model_load_error = None
except Exception as e:
    model = None
    model_load_error = str(e)

if model_load_error:
    st.error(
        f"⚠️ Could not load the model from the expected path.\n\n**Error:** {model_load_error}"
    )


# =========================================================
# CLASS NAMES
# =========================================================

# Must match training class order
class_names = [
    "Healthy",
    "Powdery",
    "Rust"
]


# =========================================================
# DISEASE INFORMATION + THEORY / SCIENCE
# =========================================================

disease_info = {

    "Healthy": {

        "status": "Healthy Plant",

        "description":
        "The AI model did not detect visual features associated "
        "with the Powdery or Rust categories.",

        "recommendation":
        "Continue regular plant care including appropriate watering, "
        "sunlight, nutrition, and routine monitoring.",

        "prevention":
        "Inspect the plant regularly and maintain good growing "
        "conditions to help reduce the risk of future diseases.",

        "causes": [
            "No pathogenic indicators detected in the leaf's visual pattern",
            "Balanced nutrition and normal chlorophyll distribution"
        ],

        "symptoms": [
            "Uniform green coloration",
            "No powdery coating or rust-colored pustules",
            "Leaf surface appears smooth and undamaged"
        ],

        "favorable_conditions":
        "Consistent watering, adequate sunlight, and good air circulation "
        "generally support continued healthy growth.",

        "color": "#2e7d32",
        "bg": "rgba(46, 125, 50, 0.12)",
        "emoji": "✅"
    },


    "Powdery": {

        "status": "Powdery Disease Category Detected",

        "description":
        "The AI model detected visual features associated with the "
        "Powdery category. Powdery mildew is commonly associated "
        "with fungal infection and may appear as white or powder-like "
        "patches on plant surfaces.",

        "recommendation":
        "Consider separating heavily affected plants where practical "
        "and removing severely affected plant material. Improving "
        "air circulation around plants may also help.",

        "prevention":
        "Avoid overcrowding plants, maintain good airflow, and "
        "regularly inspect leaves for early signs of disease.",

        "causes": [
            "Fungal species from groups such as Erysiphales, which spread "
            "via airborne spores",
            "Spores germinate readily on dry leaf surfaces, unlike many "
            "other fungi that need standing water"
        ],

        "symptoms": [
            "White to greyish powder-like coating on leaves and stems",
            "Leaf curling, yellowing, or distortion in advanced cases",
            "Reduced photosynthesis due to surface coverage"
        ],

        "favorable_conditions":
        "Powdery mildew tends to thrive in warm days, cool nights, "
        "high humidity, and shaded or poorly ventilated growing areas.",

        "color": "#e65100",
        "bg": "rgba(230, 81, 0, 0.12)",
        "emoji": "⚠️"
    },


    "Rust": {

        "status": "Rust Disease Category Detected",

        "description":
        "The AI model detected visual features associated with the "
        "Rust category. Plant rust diseases are commonly caused by "
        "fungal pathogens and may produce yellow, orange, brown, "
        "or rust-colored spots on leaves.",

        "recommendation":
        "Consider removing heavily affected leaves where appropriate. "
        "Maintain good airflow around plants and avoid unnecessary "
        "moisture remaining on foliage.",

        "prevention":
        "Monitor plants regularly, maintain suitable spacing between "
        "plants, and follow crop-specific agricultural guidance.",

        "causes": [
            "Fungal pathogens from the order Pucciniales, often requiring "
            "a living host to complete their life cycle",
            "Spores spread by wind, splashing water, and prolonged leaf "
            "wetness"
        ],

        "symptoms": [
            "Small yellow, orange, or rust-brown pustules on leaf surfaces",
            "Pustules often concentrated on the underside of leaves",
            "Premature yellowing and leaf drop in severe infections"
        ],

        "favorable_conditions":
        "Rust fungi generally favor moderate temperatures, high humidity, "
        "and extended periods of leaf surface moisture.",

        "color": "#c62828",
        "bg": "rgba(198, 40, 40, 0.12)",
        "emoji": "⚠️"
    }
}


def get_confidence_level(confidence_value):
    """Return a qualitative label and guidance text for a confidence score."""
    if confidence_value >= 85:
        return "High Confidence", "success", (
            "The model is highly confident in this prediction. The visual "
            "pattern strongly matches what it learned for this class."
        )
    elif confidence_value >= 60:
        return "Moderate Confidence", "warning", (
            "The model leans toward this prediction, but the visual "
            "pattern is less distinct. Consider verifying with a clearer "
            "or closer image of the leaf."
        )
    else:
        return "Low Confidence", "error", (
            "The model is not strongly certain about this prediction. "
            "Results at this confidence level should be treated cautiously "
            "and, if possible, re-checked with a better-quality image."
        )


# =========================================================
# MAIN CONTENT
# =========================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# LEFT COLUMN - IMAGE UPLOAD
# =========================================================

with left_column:

    st.markdown('<div class="panel-title">📤 Upload Plant Leaf</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtext">Upload a clear image of a plant leaf for AI analysis.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )

    img = None

    if uploaded_file is not None:

        img = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            img,
            caption="Uploaded Plant Leaf",
            use_container_width=True
        )


# =========================================================
# RIGHT COLUMN - PREDICTION
# =========================================================

with right_column:

    st.markdown('<div class="panel-title">🔬 Plant Health Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtext">The CNN model will analyze visual patterns in the uploaded leaf image.</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is None:

        st.info(
            "👈 Upload a plant leaf image to begin the analysis."
        )

    else:

        analyze_clicked = st.button(
            "🔍 Analyze Plant Health",
            use_container_width=True,
            type="primary",
            disabled=(model is None)
        )

        if analyze_clicked:

            with st.spinner(
                "Analyzing plant leaf..."
            ):

                # Resize image
                img_resized = img.resize(
                    (128, 128)
                )

                # Convert to array
                img_array = np.array(
                    img_resized
                ).astype("float32")

                # Normalize
                img_array = (
                    img_array / 255.0
                )

                # Add batch dimension
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Model prediction
                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                # Predicted class index
                predicted_index = int(np.argmax(
                    prediction[0]
                ))

                # Predicted class
                predicted_class = class_names[
                    predicted_index
                ]

                # Confidence
                confidence = float(np.max(
                    prediction[0]
                )) * 100

                # Full probability distribution across all classes
                probabilities = [
                    float(p) * 100 for p in prediction[0]
                ]

            st.session_state["predicted_class"] = predicted_class
            st.session_state["confidence"] = confidence
            st.session_state["probabilities"] = probabilities

        if "predicted_class" in st.session_state:

            predicted_class = st.session_state["predicted_class"]
            confidence = st.session_state["confidence"]
            info = disease_info[predicted_class]

            # =================================================
            # RESULT BADGE + METRICS
            # =================================================

            level_label, level_type, _ = get_confidence_level(confidence)

            st.markdown(
                f"""
                <span class="badge" style="background-color:{info['bg']}; color:{info['color']};">
                    {info['emoji']} {predicted_class}
                </span>
                <span class="badge" style="background-color:rgba(127,127,127,0.15); color:inherit;">
                    {level_label}
                </span>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(label="Prediction", value=predicted_class)

            with result_col2:
                st.metric(label="Confidence", value=f"{confidence:.2f}%")

            # Confidence progress (clamped to 0-100)
            st.progress(min(max(int(round(confidence)), 0), 100))

            if predicted_class == "Healthy":
                st.success("✅ HEALTHY PLANT")
            else:
                st.warning("⚠️ DISEASE CATEGORY DETECTED")


# =========================================================
# DETAILED RESULT SECTIONS (full width, below columns)
# =========================================================

if "predicted_class" in st.session_state:

    predicted_class = st.session_state["predicted_class"]
    confidence = st.session_state["confidence"]
    probabilities = st.session_state["probabilities"]
    info = disease_info[predicted_class]

    st.divider()
    st.markdown("## 📋 Detailed Analysis")

    st.markdown(
        f"""
        <div class="result-banner" style="background-color:{info['bg']}; border-color:{info['color']};">
            <h3 style="color:{info['color']};">{info['emoji']} {info['status']}</h3>
            <span class="sub" style="color:{info['color']};">Confidence: {confidence:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =================================================
    # MODEL INTELLIGENCE: FULL PROBABILITY BREAKDOWN
    # =================================================

    st.markdown("### 🧠 Model Intelligence")

    level_label, level_type, level_text = get_confidence_level(confidence)

    if level_type == "success":
        st.success(f"**{level_label}.** {level_text}")
    elif level_type == "warning":
        st.warning(f"**{level_label}.** {level_text}")
    else:
        st.error(f"**{level_label}.** {level_text}")

    prob_df = pd.DataFrame(
        {"Confidence (%)": probabilities},
        index=class_names
    )
    st.bar_chart(prob_df, color="#2e7d32")
    st.caption(
        "This chart shows the model's probability score for every class, "
        "not just the top prediction — useful for judging how close the "
        "decision was between categories."
    )

    st.divider()

    # =================================================
    # RESULT + THEORY CARDS
    # =================================================

    detail_col1, detail_col2 = st.columns(2, gap="large")

    with detail_col1:

        st.markdown(
            f"""
            <div class="card">
                <h4>🔎 About the Result</h4>
                <p>{info['description']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        symptoms_html = "".join(f"<li>{s}</li>" for s in info["symptoms"])
        st.markdown(
            f"""
            <div class="card">
                <h4>🩺 Typical Symptoms</h4>
                <ul>{symptoms_html}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                <h4>💡 Recommended Action</h4>
                <p>{info['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with detail_col2:

        causes_html = "".join(f"<li>{c}</li>" for c in info["causes"])
        st.markdown(
            f"""
            <div class="card">
                <h4>🔬 Scientific Background</h4>
                <ul>{causes_html}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                <h4>🌦️ Favorable Conditions</h4>
                <p>{info['favorable_conditions']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                <h4>🛡️ Prevention Tips</h4>
                <p>{info['prevention']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =================================================
    # DISCLAIMER
    # =================================================

    st.info(
        "ℹ️ This system is an AI-based educational prototype. "
        "It can classify images only into Healthy, Powdery, "
        "and Rust categories. Model confidence does not guarantee "
        "a correct diagnosis. For crop-specific diagnosis and "
        "treatment decisions, consult a qualified agricultural expert."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">🌿 Plant Health Prediction | CNN-Based Plant Disease Classification</div>',
    unsafe_allow_html=True
)
