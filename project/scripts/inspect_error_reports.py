import os, sys

sys.path.append(os.path.abspath("./"))
from project import create_app, app
from project.lib.datamanagement.models import ErrorReport

create_app()
app.app_context().push()
reports = ErrorReport.query.all()
for report in reports:
    print("Report ID", report.id)
    print("Image path:", report.img_path)
    print("Region index:", report.region_index)
    print("Original count:", report.original_ct)
    print("Edited count:", report.edited_ct)
    print("User ID:", report.user_id)
    print("User:", report.user)
