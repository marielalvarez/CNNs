# app.py
import streamlit as st, numpy as np, json, pickle, os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_pre
from tensorflow.keras.applications.resnet50   import preprocess_input as rs_pre
import pandas as pd 

# Descargar modelo grande si no está
def download_large_model():
    import gdown
    file_id = "1LcjEbqOLoL0I0sfa3SqMlVbj4At2NbTW"  
    output = "saved_models/resnet50.keras"
    if not os.path.exists(output):
        gdown.download(id=file_id, output=output, quiet=False)

@st.cache_resource
def load_artifacts():
    download_large_model()
    base   = tf.keras.models.load_model("saved_models/baseline.keras",   compile=False)
    mobil  = tf.keras.models.load_model("saved_models/mobilenetv2.keras",compile=False)
    resnet = tf.keras.models.load_model("saved_models/resnet50.keras",   compile=False)
    with open("saved_models/classes.json") as f:
        classes = json.load(f)
    def j(name): return json.load(open(f"saved_models/"+name+"_report.json"))
    reports = { "Baseline":j("baseline"),
                "MobileNetV2":j("mobilenetv2"),
                "ResNet50":j("resnet50")}
    return base, mobil, resnet, classes, reports

baseline, mobilenet, resnet, class_names, reports = load_artifacts()

def prep_baseline(img):
    img = img.convert("L").resize((28,28))
    arr = np.asarray(img, dtype=np.float32)/255.0
    return arr[np.newaxis, ..., np.newaxis]

def prep_rgb(img, fn):
    img = img.convert("RGB").resize((224,224))
    arr = np.asarray(img, dtype=np.float32)[np.newaxis, ...]
    return fn(arr)

st.title("🫘Bean Leaf Disease Classifier")

file = st.file_uploader("Upload a bean leaf image", type=["jpg","jpeg","png"])
show_metrics = st.checkbox("Show stored classification reports")

if file:
    img = Image.open(file)
    st.image(img, use_column_width=True)

    preds = {
        "Baseline":   baseline.predict(prep_baseline(img))[0],
        "MobileNetV2":mobilenet.predict(prep_rgb(img, mn_pre))[0],
        "ResNet50":   resnet.predict(prep_rgb(img, rs_pre))[0]
    }

    st.subheader("Predictions")
    for name, prob in preds.items():
        idx = np.argmax(prob)
        st.write(f"**{name} → {class_names[idx]}** (confidence {prob[idx]:.1%})")



    if show_metrics:
        st.subheader("Static test-set metrics")
        tabs = st.tabs(list(reports.keys()))
        for tab, (name, rep) in zip(tabs, reports.items()):
            with tab:
                df = pd.DataFrame(rep).T  

                if "accuracy" in df.index:
                    acc = df.loc["accuracy"]
                    df = df.drop("accuracy")
                    df.loc["accuracy"] = acc

                df = df.round(3)

                st.dataframe(df, use_container_width=True)