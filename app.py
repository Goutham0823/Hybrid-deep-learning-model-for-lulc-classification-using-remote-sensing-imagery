import streamlit as st
from PIL import Image
from utils.inference import predict

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Hybrid LULC & Crop Suitability System",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
"""
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}
.card {
    padding: 22px;
    border-radius: 14px;
    min-height: 220px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    animation: fadeIn 0.8s ease-in-out;
}
.lulc {
    background-color: #f4f6f8;
    color: #0b5394;
}
.cropland {
    background-color: #ecf7ee;
    color: #274e13;
}
.crop {
    background-color: #fff4e0;
    color: #7f3f00;
}
.title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 10px;
}
.text {
    font-size: 16px;
    line-height: 1.6;
    color: #000000;
}
.confidence {
    font-size: 15px;
    font-weight: 600;
    margin-top: 8px;
}
</style>
""",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>Hybrid Deep Learning System for Agriculture</h1>",
    unsafe_allow_html=True
)

st.markdown(
"""
<p style='text-align:center; font-size:18px;'>
Intelligent analysis of satellite imagery for
<b>LULC classification</b>,
<b>cropland suitability</b>, and
<b>AI-based Indian crop recommendation</b>.
</p>
""",
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# Upload Section
# --------------------------------------------------
st.subheader("Upload Satellite Image")

uploaded_file = st.file_uploader(
    "Choose a satellite image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload an image to proceed.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", width=350)

# --------------------------------------------------
# Start Analysis
# --------------------------------------------------
if st.button("Start Analysis", use_container_width=True):

    with st.spinner("Analyzing image using deep learning models..."):
        result = predict(image)

    st.success("Analysis completed successfully")

    st.markdown("---")
    st.subheader("Analysis Results")

    col1, col2, col3 = st.columns(3)

    # -------- LULC --------
    lulc_label, lulc_conf = result["lulc"]

    with col1:
        st.markdown(
            f"""
<div class="card lulc">
<div class="title">LULC Classification</div>
<div class="text"><b>{lulc_label}</b></div>
<div class="confidence">Confidence: {lulc_conf:.2f}%</div>
</div>
""",
            unsafe_allow_html=True
        )

    # -------- CROPLAND --------
    cropland_status, cropland_conf = result["cropland"]

    with col2:
        st.markdown(
            f"""
<div class="card cropland">
<div class="title">Cropland Suitability</div>
<div class="text"><b>{cropland_status}</b></div>
<div class="confidence">Confidence: {cropland_conf:.2f}%</div>
</div>
""",
            unsafe_allow_html=True
        )

    # -------- CROP RECOMMENDATION --------
    top_crops = result.get("top_crops", [])

    with col3:
        if top_crops:

            crop_html = ""
            for crop, _ in top_crops:
                crop_html += f"<div><b>{crop}</b></div>"

            st.markdown(
                f"""
<div class="card crop">
<div class="title">Crop Recommendation</div>
<div class="text">{crop_html}</div>
</div>
""",
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                """
<div class="card crop">
<div class="title">Crop Recommendation</div>
<div class="text"><b>Not suitable for cropping</b></div>
</div>
""",
                unsafe_allow_html=True
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Hybrid AI System • EuroSAT Dataset • AI-based Crop Advisory</p>",
    unsafe_allow_html=True
)