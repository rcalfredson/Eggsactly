from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
import os
# db = SQLAlchemy()

sessions = {}
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tif"}
NET_ARCH = ("fcrn", "splinedist")[1]

def create_app():
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.secret_key = os.urandom(24)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # db.init_app(app)
    return app