# Embryo Grading AI — Web Developer Deployment Package

Welcome! This folder contains everything you need to build and deploy the **Embryo Grading & Viability Assessment Web Application**.

---

## 📁 Package Contents

```text
embryo_web_deployment/
├── models/                               # All 4 Pretrained Checkpoint Files
│   ├── best_convnext_tiny.pth
│   ├── best_densenet121.pth
│   ├── best_proper_efficientnet_b0.pth
│   └── best_resnet50_full.pth
├── preprocessing_pipeline.py             # Image Preprocessing Library (Crop, CLAHE, Sharpness)
├── inference_pipeline.py                 # Ready-to-Use Prediction Engine (Super-Ensemble)
├── About_Embryo_Grading_Project.pdf      # Detailed Project Background & Clinical Motivation
├── Embryo_Model_Evaluation_Metrics.pdf   # Complete Benchmarks & Mathematical Zero-Leakage Proof
├── Embryo_Image_Preprocessing_Pipeline.pdf # Step-by-Step Guide to the Image Pipeline
└── README_FOR_WEBSITE_DEVELOPER.md       # This Guide
```

---

## 🚀 Quick-Start Code for Web Backend

You can run predictions with **3 lines of Python**:

```python
from inference_pipeline import EmbryoPredictor

# 1. Initialize predictor (loads all 4 models once on startup)
predictor = EmbryoPredictor()

# 2. Predict any uploaded embryo image
result = predictor.predict("path_to_uploaded_image.png")

# 3. Use the result
print(result)
```

### 📦 Example Output Format (JSON):
```json
{
  "predicted_grade": "Grade-1",
  "quality_tier": "Good Quality Blastocyst",
  "clinical_viability": "Viable / Transferable (Recommended for Transfer / Freeze)",
  "confidence_score": 0.8842,
  "confidence_percentage": "88.4%",
  "probabilities": {
    "Grade-1": 0.8842,
    "Grade-2": 0.1015,
    "Grade-3": 0.0143
  },
  "clinical_recommendation": "Optimal morphology (Gardner AA/AB equivalent). Priority candidate for single blastocyst transfer (eSET) or high-survival vitrification."
}
```

---

## 🌐 Sample Integration: FastAPI Endpoint

```python
from fastapi import FastAPI, UploadFile, File
from inference_pipeline import EmbryoPredictor
import shutil
import os

app = FastAPI(title="AI Embryo Grading API")
predictor = EmbryoPredictor()

@app.post("/api/grade")
async def grade_embryo(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = predictor.predict(temp_path)
    os.remove(temp_path)
    return result
```

---

## 🎨 Sample Integration: Streamlit App

```python
import streamlit as st
from PIL import Image
from inference_pipeline import EmbryoPredictor

st.title("🔬 AI Embryo Quality Assessment & Grading")

@st.cache_resource
def load_model():
    return EmbryoPredictor()

predictor = load_model()

uploaded_file = st.file_uploader("Upload Embryo Microscope Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Embryo", width=300)
    
    with st.spinner("Analyzing morphology..."):
        result = predictor.predict(img)
    
    st.success(f"**Predicted Grade:** {result['predicted_grade']} ({result['quality_tier']})")
    st.info(f"**Viability Status:** {result['clinical_viability']}")
    st.metric("Model Confidence", result['confidence_percentage'])
    st.write("**Probabilities:**", result['probabilities'])
    st.write("**Clinical Note:**", result['clinical_recommendation'])
```

---

## 🔬 Multi-Slice / Z-Stack Feature (Bonus)
If a user uploads multiple focal slices for a single embryo, you can automatically pick the sharpest in-focus slice:

```python
from preprocessing_pipeline import select_best_slice

uploaded_slices = ["slice_01.png", "slice_02.png", "slice_03.png"]
best_slice_path = select_best_slice(uploaded_slices)
result = predictor.predict(best_slice_path)
```

---

## 🛡️ Medical Validation & Data Leakage Note
* **Zero Patient Leakage:** The model was tested on 122 completely unseen patients with strict group segregation.
* **Accuracy:** Reaches **81.15% 3-class accuracy** in Patient In-Focus Mode and **84.31% – 86.07% clinical viability accuracy** (ROC-AUC: 92.34%).
* **Zero Severe Errors:** Grade-1 blastocysts were never misclassified as Grade-3.
