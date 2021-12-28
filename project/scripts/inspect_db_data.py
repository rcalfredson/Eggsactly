import argparse
import os, sys

sys.path.append(os.path.abspath("./"))
from project import create_app, app
from project.lib.datamanagement.models import (
    EggLayingImage,
    EggRegionTemplate,
    ErrorReport,
)

p = argparse.ArgumentParser(
    description="look up a user's name and email via" + " their ID"
)
p.add_argument(
    "model_type",
    help="model whose table should be shown (not case-sensitive)",
    type=str,
)
opts = p.parse_args()

create_app()
app.app_context().push()
model_type = {
    "egglayingimage": EggLayingImage,
    "eggregiontemplate": EggRegionTemplate,
    "errorreport": ErrorReport,
}[opts.model_type.lower()]
entities = model_type.query.all()
for ent in entities:
    if model_type == EggLayingImage:
        print('image data:', ent.image)
    elif model_type == EggRegionTemplate:
        print("Template ID:", ent.id)
        print("Template name:", ent.name)
        print("Template data:", ent.data)
        print("User ID:", ent.user_id)
        print("User:", ent.user)
    elif model_type == ErrorReport:
        print("Report ID", ent.id)
        print("Image path:", ent.img_path)
        print("Region index:", ent.region_index)
        print("Original count:", ent.original_ct)
        print("Edited count:", ent.edited_ct)
        print("User ID:", ent.user_id)
        print("User:", ent.user)
