import os
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import hashlib
import pandas as pd  

_real_md5 = hashlib.md5
def _patched_md5(data=b"", *args, **kwargs):
    return _real_md5(data)
hashlib.md5 = _patched_md5

UPLOAD_FOLDER = "static/data"
REPORT_FOLDER = "static/reports"
MODEL_JSON = "model/model.json"
MODEL_WEIGHTS = "model/model.h5"
ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER


def load_skin_data_from_csv(csv_path="data/disease_info.csv"):
    df = pd.read_csv(csv_path)
    skin_data = {}
    for _, row in df.iterrows():
        skin_data[int(row["id"])] = {
            "name": row["name"],
            "symptoms": [s.strip() for s in str(row["symptoms"]).split(";")],
            "solution": [s.strip() for s in str(row["solution"]).split(";")]
        }
    return skin_data

SKIN_DATA = load_skin_data_from_csv()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def generate_pdf(patient_name, report_filename, image_path, disease_info, pred_probs):
    file_path = os.path.join(app.config["REPORT_FOLDER"], report_filename)
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Skin Lesion Diagnosis Report")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Patient Name: {patient_name}")
    c.drawString(350, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Predicted Disease:")
    c.setFont("Helvetica", 11)
    c.drawString(margin+150, y, disease_info["name"])
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Class Probabilities:")
    y -= 18
    c.setFont("Helvetica", 10)

    for cls, prob in pred_probs:
        c.drawString(margin+15, y, f"- {cls}: {prob:.2f}%")
        y -= 14

   
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Symptoms:")
    y -= 18
    c.setFont("Helvetica", 10)
    for s in disease_info["symptoms"]:
        c.drawString(margin+15, y, f"- {s}")
        y -= 14

    
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Suggested Treatment:")
    y -= 18
    c.setFont("Helvetica", 10)
    for s in disease_info["solution"]:
        c.drawString(margin+15, y, f"- {s}")
        y -= 14

    
    if os.path.exists(image_path):
        c.showPage()
        c.drawImage(image_path, 150, 200, width=300, height=300)

    c.save()
    return file_path


def load_model_from_files():
    with open(MODEL_JSON) as f:
        model = model_from_json(f.read())
    model.load_weights(MODEL_WEIGHTS)
    return model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploaded", methods=["POST"])
def uploaded():
    file = request.files.get("file")
    patient_name = request.form.get("name", "Unknown")

    if not file or not allowed_file(file.filename):
        return "Invalid file", 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(f"{timestamp}_{file.filename}")
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    model = load_model_from_files()
    img = load_img(file_path, target_size=(224,224))
    img = np.expand_dims(img_to_array(img)/255.0, 0)

    preds = model.predict(img)[0]
    pred_idx = np.argmax(preds)
    pred_percent = preds[pred_idx] * 100

    pred_probs = [(SKIN_DATA[i]["name"], preds[i] * 100) for i in range(len(preds))]
    disease_info = SKIN_DATA[pred_idx]

    report_name = f"report_{patient_name}_{timestamp}.pdf"
    generate_pdf(patient_name, report_name, file_path, disease_info, pred_probs)

    K.clear_session()

    return render_template(
        "uploaded.html",
        img_file=filename,
        disease=disease_info["name"],
        accuracy=round(pred_percent,2),
        symptoms=disease_info["symptoms"],
        solution=disease_info["solution"],
        prob_list=[{"name": n, "prob": round(p,2)} for n,p in pred_probs],
        report_file=report_name,
        patient_name=patient_name
    )

@app.route("/reports/<filename>")
def reports(filename):
    return send_from_directory(app.config["REPORT_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
