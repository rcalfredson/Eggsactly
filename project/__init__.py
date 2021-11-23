from flask import Flask
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os

db = SQLAlchemy()
app = Flask(__name__)
socketIO = SocketIO(app)

sessions = {}
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tif"}
NET_ARCH = ("fcrn", "splinedist")[1]


def create_app():
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.secret_key = os.urandom(24)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)
    from .lib.datamanagement.models import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in
        # the query for the user
        return User.query.get(int(user_id))

    app.socketIO = socketIO
    from project.routes.auth import auth as auth_blueprint
    from project.routes.main import main as main_blueprint

    app.register_blueprint(main_blueprint)
    app.register_blueprint(auth_blueprint)
    return app
