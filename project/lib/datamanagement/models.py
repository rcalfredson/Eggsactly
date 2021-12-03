from flask_login import UserMixin
from flask_dance.contrib.google import google
from flask_login import login_user, current_user, login_required, logout_user
from werkzeug.security import generate_password_hash
import sqlalchemy
import oauthlib
import os
from ... import db, app


class User(UserMixin, db.Model):
    id = db.Column(
        db.Integer, primary_key=True
    )  # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))
    is_google = db.Column(db.Boolean())
    is_local = db.Column(db.Boolean())


def login_google_user():
    user_info_endpoint = "/oauth2/v2/userinfo"
    google_data, name = None, None
    if google.authorized:
        try:
            google_data = google.get(user_info_endpoint).json()
            name = google_data["name"]
            if not current_user.is_authenticated:
                user = User.query.filter_by(
                    email=google_data["email"], is_google=True, is_local=False
                ).first()
                if not user:
                    user = User(
                        email=google_data["email"],
                        name=google_data["name"],
                        password=generate_password_hash(
                            str(os.urandom(24)), method="sha256"
                        ),
                        is_google=True,
                        is_local=False,
                    )

                    db.session.add(user)
                    try:
                        db.session.commit()
                    except sqlalchemy.exc.IntegrityError:
                        db.session.rollback()
                        user = User.query.filter_by(
                            email=google_data["email"], is_google=False, is_local=False
                        ).first()
                login_user(user, remember=False)
        except oauthlib.oauth2.rfc6749.errors.TokenExpiredError:
            del app.blueprints["google"].token
    return {"data": google_data, "name": name}
