import streamlit as st
import os, time
from pathlib import Path
from PIL import Image
import torch

st.set_page_config(page_title="🦺 PPE Detection App", layout="centered")
st.title("🦺 PPE Detection App")
st.write("Upload an image to detect helmets, vests, masks, and more!")

# --------- Model loader (Cloud-safe) ----------
@st.cache_resource
def load_model():
    root = Path(__file__).resolve().parent
    candidates = [
        root / "weights" / "best.pt",   # your LFS-trained weights
        root / "best.pt",               # optional fallback
        root / "yolov5s.pt",            # tiny demo fallback
    ]
    weight = next((str(p) for p in candidates if p.exists()), None)
    if not weight:
        st.error("No weights found. Add your trained file at `weights/best.pt` "
                 "or keep `yolov5s.pt` in the repo root.")
        st.stop()
    return torch.hub.load('ultralytics/yolov5', 'custom', path=weight, force_reload=False)

model = load_model()

# --------- App -----------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image to /tmp (works locally and on Streamlit Cloud)
    img = Image.open(uploaded_file).convert("RGB")
    img_path = Path("/tmp") / f"temp_{int(time.time())}_{uploaded_file.name}"
    img.save(img_path)

    st.image(img, caption="📸 Uploaded Image", use_container_width=True)
    st.write("🔍 Detecting...")

    # Create a unique folder under /tmp for each detection
    results_dir = Path("/tmp") / f"detection_{int(time.time())}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run detection and save annotated image(s)
    results = model(str(img_path))
    results.save(save_dir=str(results_dir))

    # Find the latest detection image
    detected_files = sorted(
        [results_dir / f for f in os.listdir(results_dir)],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    detected_img_path = detected_files[0] if detected_files else None
    if detected_img_path:
        st.image(str(detected_img_path), caption="🧠 Detection Results", use_container_width=True)

    # Extract detection info
    detected_objects = results.pandas().xyxy[0]

    if len(detected_objects) == 0:
        st.warning("⚪ No objects detected.")
    else:
        st.subheader("📋 Detection Details")

        color_map = {
            "Hardhat": "green", "Mask": "green", "Safety Vest": "green",
            "NO-Hardhat": "red", "NO-Mask": "red", "NO-Safety Vest": "red",
            "Person": "yellow", "Safety Cone": "yellow", "Machinery": "yellow", "Vehicle": "yellow"
        }

        sections = {"🟢 Safe Equipment": [], "🟡 Other Objects": [], "🔴 Unsafe Conditions": []}
        for _, row in detected_objects.iterrows():
            label = row["name"]; conf = row["confidence"]
            color = color_map.get(label, "white")
            text = f"<span style='color:{color}; font-size:18px;'><b>{label}</b> — Confidence: {conf:.2f}</span>"
            if label.startswith("NO-"):
                sections["🔴 Unsafe Conditions"].append(text)
            elif label in ["Hardhat", "Mask", "Safety Vest"]:
                sections["🟢 Safe Equipment"].append(text)
            else:
                sections["🟡 Other Objects"].append(text)

        for title, items in sections.items():
            if items:
                st.markdown(f"### {title}")
                for t in items:
                    st.markdown(t, unsafe_allow_html=True)

        st.write("---")
        st.subheader("🧾 Detection Summary")
        counts = detected_objects["name"].value_counts()
        for obj, count in counts.items():
            color = color_map.get(obj, "white")
            st.markdown(f"<span style='color:{color}; font-size:16px;'>• {obj}: {count}</span>",
                        unsafe_allow_html=True)

    # Cleanup temporary file
    try:
        img_path.unlink(missing_ok=True)  # Python 3.8+: use os.remove if older
    except Exception:
        pass
