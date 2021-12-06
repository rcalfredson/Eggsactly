import argparse
import os, sys

sys.path.append(os.path.abspath("./"))
from project import db, create_app, app
from project.lib.datamanagement.models import User

create_app()
app.app_context().push()
p = argparse.ArgumentParser(
    description="look up a user's name and email via" + " their ID"
)
p.add_argument("user_id", help="ID of the user to look up", type=int)
opts = p.parse_args()
queried_user = User.query.get(opts.user_id)
print(f"ID: {opts.user_id}\tName: {queried_user.name}\tEmail: {queried_user.email}")
print(
    f"Is a Google-based user: {queried_user.is_google}"
    f"\tIs a local user: {queried_user.is_local}"
)
