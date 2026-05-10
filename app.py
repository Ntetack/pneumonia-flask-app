from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

import os
import torch
import numpy as np

from PIL import Image

import torchvision.transforms as transforms

from model import PneumoniaCNN

# =========================
# FLASK CONFIG
# =========================
app = Flask(__name__)

app.secret_key = "secretkey"

app.config['UPLOAD_FOLDER'] = 'static/uploads'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'

# Create folders if they don't exist
os.makedirs("database", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

db = SQLAlchemy(app)

# =========================
# DATABASE
# =========================
class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    filename = db.Column(db.String(200))

    result = db.Column(db.String(50))


class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )


# =========================
# LOAD MODEL
# =========================
device = torch.device("cpu")

model = PneumoniaCNN(num_classes=3)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.eval()

# =========================
# CLASSES
# =========================
classes = {
    0: "BACTERIAL",
    1: "NORMAL",
    2: "VIRAL"
}

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOGIN
# =========================

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        user = User.query.filter_by(
            username=username
        ).first()

        if user and check_password_hash(user.password, password):

            session["user"] = username

            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid credentials"
        )

    return render_template("login.html")




@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        existing_user = User.query.filter_by(
            username=username
        ).first()

        if existing_user:

            return render_template(
                "register.html",
                error="Username already exists"
            )

        hashed_password = generate_password_hash(password)

        new_user = User(
            username=username,
            password=hashed_password
        )

        db.session.add(new_user)

        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


# =========================
# DASHBOARD
# =========================
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    if "user" not in session:

        return redirect(url_for("login"))

    if request.method == "POST":

        if "image" not in request.files:

            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":

            return redirect(request.url)

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(filepath)

        # =========================
        # IMAGE PREPROCESSING
        # =========================
        image = Image.open(filepath).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        # =========================
        # PREDICTION
        # =========================
        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probabilities, 1)

        result = classes[predicted.item()]

        confidence = confidence.item() * 100

        # =========================
        # SAVE DATABASE
        # =========================
        new_prediction = Prediction(
            filename=filename,
            result=result
        )

        db.session.add(new_prediction)

        db.session.commit()

        return render_template(
            "result.html",
            filename=filename,
            result=result,
            confidence=round(confidence, 2)
        )

    return render_template("dashboard.html")

# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("login"))

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    app.run(debug=True)