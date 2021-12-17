from dotenv import load_dotenv
from flask import Flask
from flask_dance.contrib.google import make_google_blueprint
from flask_socketio import SocketIO
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os
import warnings

db = SQLAlchemy()
app = Flask(__name__)
login_manager = LoginManager()
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
socketIO = SocketIO(app, manage_session=False)

sessions = {}
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tif"}
NET_ARCH = ("fcrn", "splinedist")[1]


def create_app():
    load_dotenv()
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    # app.secret_key = os.urandom(24)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.google_client_id = os.getenv("GOOGLE_CLIENT_ID")
    app.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    app.secret_key = os.getenv("secret_key")
    db.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)
    from .lib.datamanagement.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    app.socketIO = socketIO
    from project.routes.auth import auth as auth_blueprint
    from project.routes.main import main as main_blueprint
    from project.routes.tasks import tasks as tasks_blueprint

    google_blueprint = make_google_blueprint(
        client_id=app.google_client_id,
        client_secret=app.google_client_secret,
        reprompt_consent=True,
        scope=["profile", "email"],
    )

    app.register_blueprint(main_blueprint)
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(tasks_blueprint)
    app.register_blueprint(google_blueprint, url_prefix="/login")
    return app
