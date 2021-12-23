from dotenv import load_dotenv
from flask import Flask
from flask_dance.contrib.google import make_google_blueprint
from flask_login import LoginManager
from flask_session import Session
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
import os
import warnings

from project.lib.web.backend_types import BackendTypes

load_dotenv()
db = SQLAlchemy()
app = Flask(__name__)
login_manager = LoginManager()
backend_type = BackendTypes[os.getenv("EGG_COUNTING_BACKEND_TYPE")]
print("backend type:", backend_type)
if backend_type == BackendTypes.gcp:
    # sql_addr = (
    #     f"mysql://root:{os.environ['GOOGLE_SQL_DB_PASSWORD']}@"
    #     f"{os.environ['GOOGLE_SQL_DB_PVT_IP']}/data"
    # )
    sql_addr = "sqlite:///db.sqlite"
    flask_session_type = "sqlalchemy"
elif backend_type == BackendTypes.local:
    sql_addr = "sqlite:///db.sqlite"
    flask_session_type = "filesystem"
app.config["BACKEND_TYPE"] = backend_type
app.config["SQLALCHEMY_DATABASE_URI"] = sql_addr
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
app.config["SESSION_TYPE"] = flask_session_type
session = Session(app)
if flask_session_type == "sqlalchemy":
    session.app.session_interface.db.create_all()
socketIO = SocketIO(app, manage_session=False)

sessions = {}
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tif"}


def create_app():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.google_client_id = os.getenv("GOOGLE_CLIENT_ID")
    app.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    app.secret_key = os.getenv("secret_key")
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
