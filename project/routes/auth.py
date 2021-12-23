from flask import Blueprint, flash, redirect, request, session, url_for
from flask_dance.contrib.google import google
from flask_login import current_user, login_required, login_user, logout_user
import json
import os
from werkzeug.security import check_password_hash, generate_password_hash

from project import app, db
from project.lib.datamanagement.models import User, login_google_user
from project.lib.users import users

auth = Blueprint("auth", __name__)
with open("project/configs/privileged_ips.json") as f:
    privileged_ips = json.load(f)


@auth.route("/check_request_origin", methods=["POST"])
def check_request_origin():
    ip_addr = request.form.get("ip")
    if "." in ip_addr:
        local = ip_addr in privileged_ips["v4"]
    elif ":" in ip_addr:
        network = ":".join(ip_addr.split(":")[:4])
        local = network in privileged_ips["v6"]
    else:
        local = False
    session["local-ip"] = local
    return {"local": local}


@auth.route("/login/local", methods=["POST"])
def login_local_user():
    if not session["local-ip"]:
        return {"success": False, "message": "IP address not verified"}
    name = request.form.get("name")
    user = User.query.filter_by(name=name, is_local=True).first()
    if not user:
        user = User(
            email=str(os.urandom(24)),
            name=name,
            password=generate_password_hash(str(os.urandom(24)), method="sha256"),
            is_local=True,
            is_google=False,
        )
        db.session.add(user)
        db.session.commit()
    login_user(user, remember=False)
    return {"success": True}


@auth.route("/users/local")
def get_local_users():
    return {"users": users if session["local-ip"] else []}


@auth.route("/login/oauth/google")
def login_oath_google():
    if hasattr(current_user, "google_login"):
        delattr(current_user, "google_login")
    redirect(url_for("google.login"))


@auth.route("/user/exists/google/<email>")
def user_exists(email):
    user = User.query.filter_by(email=email, is_google=True, is_local=False).first()
    return {"userExists": True if user else False}


@auth.route("/user/info/google")
def user_info():
    keys = ("name", "id", "picture")
    user_info_endpoint = "/oauth2/v2/userinfo"
    if not google.authorized:
        return {k: None for k in keys}
    google_data = google.get(user_info_endpoint).json()
    login_google_user()
    return {k: google_data[k] for k in keys}


@auth.route("/login", methods=["POST"])
def login_post():
    fail_message = "Please check your login details and try again."
    email = request.form.get("email")
    password = request.form.get("password")
    modal = True if request.form.get("modal") else False
    remember = True if request.form.get("remember") else False

    user = User.query.filter_by(email=email, is_local=False).first()

    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it to the hashed password in the database
    if not user or not (user.is_google or check_password_hash(user.password, password)):
        flash(fail_message)
        return (
            {"success": False, "id": None, "name": None, "message": fail_message}
            if modal
            else redirect(url_for("auth.login"))
        )  # if the user doesn't exist or password is wrong, reload the page
    elif user.is_google:
        return (
            {"success": False, "message": "has_google"}
            if modal
            else redirect(url_for("google.login"))
        )
    login_user(user, remember=remember)
    return (
        {"success": True, "id": user.id, "name": user.name}
        if modal
        else redirect(url_for("main.index"))
    )


@auth.route("/signup", methods=["POST"])
def signup_post():
    modal = True if request.form.get("modal") else False
    error_msg = "This email address is already registered. "
    email = request.form.get("email")
    name = request.form.get("name")
    password = request.form.get("password")

    user = User.query.filter_by(email=email, is_local=False).first()
    if user:
        if user.is_google:
            error_msg += 'Try clicking "Sign in with Google" from the login menu.'
        else:
            error_msg += "Try logging in, or create an account with a different email."
        flash(error_msg)
        return (
            {"success": False, "message": error_msg}
            if modal
            else redirect(url_for("auth.signup"))
        )

    # create a new user with the form data. Hash the password so the plaintext version isn't saved.
    new_user = User(
        email=email,
        name=name,
        password=generate_password_hash(password, method="sha256"),
        is_google=False,
        is_local=False,
    )

    # add the new user to the database
    db.session.add(new_user)
    db.session.commit()
    return {"success": True} if modal else redirect(url_for("auth.login"))


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    kwargs = {}
    if request.args.get("origin"):
        kwargs = {"origin": request.args.get("origin")}
    if app.blueprints["google"].token:
        del app.blueprints["google"].token
    return redirect(url_for("main.index", **kwargs))
