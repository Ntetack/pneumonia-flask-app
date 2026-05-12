from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from sqlalchemy import func
from datetime import datetime, timedelta

import os
import torch
import numpy as np

from datetime import datetime

from PIL import Image

import torchvision.transforms as transforms

from model import PneumoniaCNN

# APP CONFIG
app = Flask(__name__)

app.secret_key = "secretkey"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'

db = SQLAlchemy(app)

# LOGIN REQUIRED DECORATOR
def login_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):

        if "user" not in session:
            return redirect(url_for("login"))

        return f(*args, **kwargs)

    return decorated_function



os.makedirs("static/uploads", exist_ok=True)

# DATABASE MODELS
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

class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    filename = db.Column(db.String(200))

    result = db.Column(db.String(50))

    confidence = db.Column(db.Float)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id')
    )

# LOAD MODEL
device = torch.device("cpu")

model = PneumoniaCNN(num_classes=3)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

classes = {
    0: "BACTERIAL",
    1: "NORMAL",
    2: "VIRAL"
}

# IMAGE TRANSFORM
transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# LOGIN
@app.route("/", methods=["GET", "POST"])
def login():

    if "user" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        user = User.query.filter_by(
            username=username
        ).first()

        if user and check_password_hash(user.password, password):

            session["user"] = user.id

            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid credentials"
        )

    return render_template("login.html")

# REGISTER
@app.route("/register", methods=["GET", "POST"])
def register():

    if "user" in session:
        return redirect(url_for("dashboard"))

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

# DASHBOARD
@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():

    if "user" not in session:

        return redirect(url_for("login"))

    if request.method == "POST":

        file = request.files["image"]

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            "static/uploads",
            filename
        )

        file.save(filepath)

        image = Image.open(filepath).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        image = image.to(device)

        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        result = classes[predicted.item()]

        confidence = confidence.item() * 100

        prediction = Prediction(
            filename=filename,
            result=result,
            confidence=round(confidence, 2),
            user_id=session["user"]
        )

        db.session.add(prediction)

        db.session.commit()

        return render_template(
            "result.html",
            filename=filename,
            result=result,
            confidence=round(confidence, 2)
        )

    return render_template("dashboard.html")

# ANALYTICS
@app.route("/analytics")
@login_required
def analytics():

    filter_type = request.args.get("filter", "all")

    predictions_query = Prediction.query

    today = datetime.utcnow()

    if filter_type == "today":

        predictions_query = predictions_query.filter(
            Prediction.created_at >= today - timedelta(days=1)
        )

    elif filter_type == "week":

        predictions_query = predictions_query.filter(
            Prediction.created_at >= today - timedelta(days=7)
        )

    elif filter_type == "month":

        predictions_query = predictions_query.filter(
            Prediction.created_at >= today - timedelta(days=30)
        )

    predictions = predictions_query.all()

    total = len(predictions)

    normal = len([
        p for p in predictions
        if p.result == "NORMAL"
    ])

    bacterial = len([
        p for p in predictions
        if p.result == "BACTERIAL"
    ])

    viral = len([
        p for p in predictions
        if p.result == "VIRAL"
    ])

    infected = bacterial + viral

    infection_rate = 0

    if total > 0:

        infection_rate = round(
            (infected / total) * 100,
            2
        )

        # LINE CHART DATA
    
    chart_data = db.session.query(

        func.date(Prediction.created_at),

        func.count(Prediction.id)

    ).group_by(

        func.date(Prediction.created_at)

    ).all()

    line_labels = [
        str(item[0])
        for item in chart_data
    ]

    line_values = [
        item[1]
        for item in chart_data
    ]

        # INFECTION RATE CHART
    
    daily_predictions = db.session.query(
        func.date(Prediction.created_at),
        Prediction.result
    ).all()

    infection_data = {}

    for date, result in daily_predictions:

        date = str(date)

        if date not in infection_data:

            infection_data[date] = {
                "total": 0,
                "infected": 0
            }

        infection_data[date]["total"] += 1

        if result != "NORMAL":

            infection_data[date]["infected"] += 1

    infection_labels = []

    infection_rates = []

    for date in sorted(infection_data.keys()):

        total_cases = infection_data[date]["total"]

        infected_cases = infection_data[date]["infected"]

        rate = 0

        if total_cases > 0:

            rate = round(
                (infected_cases / total_cases) * 100,
                2
            )

        infection_labels.append(date)

        infection_rates.append(rate)

        # RECENT PREDICTIONS
    
    recent_predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).limit(5)

    return render_template(

        "analytics.html",

        total=total,

        normal=normal,

        bacterial=bacterial,

        viral=viral,

        infected=infected,

        infection_rate=infection_rate,

        recent_predictions=recent_predictions,

        line_labels=line_labels,

        line_values=line_values,

        infection_labels=infection_labels,

        infection_rates=infection_rates,

        current_filter=filter_type
    )


# HISTORY
@app.route("/history")
@login_required
def history():

    predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        predictions=predictions
    )

# LOGOUT
@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("login"))

# MAIN
if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    app.run(debug=True)