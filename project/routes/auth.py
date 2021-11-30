from flask import Blueprint, flash, render_template, redirect, url_for, request
from flask_dance.contrib.google import google
from flask_login import login_user, current_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from ..lib.datamanagement.models import User, login_google_user
from .. import db, app

auth = Blueprint("auth", __name__)


@auth.route("/login")
def login():
    return render_template("login.html")


@auth.route("/login/oauth/google")
def login_oath_google():
    if hasattr(current_user, "google_login"):
        delattr(current_user, "google_login")
    redirect(url_for("google.login"))


@auth.route("/user/exists/google/<email>")
def user_exists(email):
    user = User.query.filter_by(email=email, is_google=True).first()
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

    user = User.query.filter_by(email=email).first()

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


@auth.route("/signup")
def signup():
    return render_template("signup.html")


@auth.route("/signup", methods=["POST"])
def signup_post():
    modal = True if request.form.get("modal") else False
    error_msg = "This email address is already registered. "
    email = request.form.get("email")
    name = request.form.get("name")
    password = request.form.get("password")

    user = User.query.filter_by(email=email).first()
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
    )

    # add the new user to the database
    db.session.add(new_user)
    db.session.commit()
    return {"success": True} if modal else redirect(url_for("auth.login"))


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    if app.blueprints["google"].token:
        del app.blueprints["google"].token
    return redirect(url_for("main.index"))
