from datetime import datetime, timedelta
from flask_dance.contrib.google import google
from flask_login import current_user, login_user, UserMixin
import oauthlib
import os
import sqlalchemy
from werkzeug.security import generate_password_hash

from ... import app, db

LONGBLOG_LEN = (2 ** 32) - 1


def delete_expired_rows(cls):
    # adapted from code at this source:
    # https://silvaneves.org/deleting-old-items-in-sqlalchemy.html

    expiration_seconds = 60 * 60
    limit = datetime.utcnow() - timedelta(seconds=expiration_seconds)
    query_results = cls.query.filter(cls.timestamp <= limit)
    for item in query_results:
        db.session.delete(item)
    db.session.commit()


class User(UserMixin, db.Model):
    id = db.Column(
        db.Integer, primary_key=True
    )  # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(1000), nullable=False)
    is_google = db.Column(db.Boolean, nullable=False)
    is_local = db.Column(db.Boolean, nullable=False)


class SocketIOUser(db.Model):
    id = db.Column(db.String(24), primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class ErrorReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.LargeBinary(length=LONGBLOG_LEN), nullable=False)
    outline_image = db.Column(db.LargeBinary(length=LONGBLOG_LEN), nullable=False)
    img_path = db.Column(db.String(1000), nullable=False)
    region_index = db.Column(db.Integer, nullable=False)
    original_ct = db.Column(db.Integer, nullable=False)
    edited_ct = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    user = db.relationship(
        "User", backref=db.backref("error_reports", lazy=True, cascade="all,delete")
    )
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    egg_counting_model_id = db.Column(db.String(1000), nullable=False)


class EggRegionTemplate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.JSON, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(1000), nullable=False)
    user = db.relationship(
        "User", backref=db.backref("templates", lazy=True, cascade="all,delete")
    )


class EggLayingImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.String(24), db.ForeignKey(SocketIOUser.id), nullable=False
    )
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    image = db.Column(db.LargeBinary(length=LONGBLOG_LEN), nullable=False)
    annotated_img = db.Column(db.LargeBinary(length=LONGBLOG_LEN), nullable=True)
    basename = db.Column(db.String(1000), nullable=False)
    user = db.relationship(
        "SocketIOUser", backref=db.backref("images", lazy=True, cascade="all,delete")
    )


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
