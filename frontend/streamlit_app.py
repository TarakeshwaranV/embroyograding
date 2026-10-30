from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from predict import load_efficientnet, predict_image


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = PROJECT_ROOT / "efficientnet_embryo_best.pth"
GRADES = ("Grade-1 (Good)", "Grade-2 (Fair)", "Grade-3 (Poor)")


st.set_page_config(page_title="Embryo Grading", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f4f6f2; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1120px; padding-top: 2rem; }
    h1, h2, h3 { color: #173d37; }
    [data-testid="stFileUploader"] section { background: #fff; border: 1px dashed #82988e; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(checkpoint_path, device_name):
    return load_efficientnet(weights_path=checkpoint_path, device=device_name)


st.title("Embryo Image Grading")
st.caption("Upload blastocyst images to estimate a quality grade.")
st.warning(
    " Predictions are not clinically calibrated and are not a diagnosis."
)

if not CHECKPOINT.is_file():
    st.error(f"Model checkpoint not found: {CHECKPOINT.name}")
    st.stop()

uploads = st.file_uploader(
    "Choose one or more images",
    type=("png", "jpg", "jpeg", "bmp", "tif", "tiff"),
    accept_multiple_files=True,
)

if uploads:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with st.spinner("Loading grading model..."):
            model = get_model(str(CHECKPOINT), device)
    except Exception as exc:
        st.error(f"Could not load the grading model: {exc}")
        st.stop()

    st.subheader(f"Results · {len(uploads)} image{'s' if len(uploads) != 1 else ''}")
    for upload in uploads:
        with st.container(border=True):
            image_column, result_column = st.columns((1, 1.5), gap="large")
            try:
                image = Image.open(BytesIO(upload.getvalue())).convert("RGB")
                with torch.inference_mode():
                    grade, probabilities = predict_image(image, model, device=device, use_tta=True)
            except Exception as exc:
                st.error(f"Could not process {upload.name}: {exc}")
                continue

            with image_column:
                st.image(image, caption=upload.name, use_container_width=True)
            with result_column:
                confidence = float(probabilities[int(grade[-1]) - 1])
                st.metric("Predicted grade", grade, f"{confidence:.1%} model confidence")
                for label, probability in zip(GRADES, probabilities):
                    st.write(f"{label} · {float(probability):.1%}")
                    st.progress(float(probability))