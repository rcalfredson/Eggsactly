import argparse
import os, sys

sys.path.append(os.path.abspath("./"))
from project import create_app, app
from project.lib.datamanagement.models import (
    EggLayingImage,
    EggRegionTemplate,
    ErrorReport,
)

query_suffix = (
    ", combined with any other inputted query parameters relevant to"
    + " this model type. If included, it overrides the value inputted for model_type."
)
p = argparse.ArgumentParser(
    description="look up a user's name and email via" + " their ID"
)
p.add_argument(
    "model_type",
    help="model whose table should be shown (not case-sensitive)",
    type=str,
)
p.add_argument(
    "--session_id",
    help="runs an EggLayingImage query for a given session ID" + query_suffix,
)
p.add_argument(
    "--basename", help="runs an EggLaying query for a given basename" + query_suffix
)
opts = p.parse_args()

create_app()
app.app_context().push()
model_type = {
    "egglayingimage": EggLayingImage,
    "eggregiontemplate": EggRegionTemplate,
    "errorreport": ErrorReport,
}[opts.model_type.lower()]
# entities = model_type.query.all()
if opts.session_id or opts.basename:
    kwargs = {k: getattr(opts, k) for k in ('session_id', 'basename')}
    entities = EggLayingImage.query.filter_by(**kwargs)
else:
    entities = model_type.query.all()
print('Number of entities:', len(entities))
for ent in entities:
    if model_type == EggLayingImage:
        print("Image ID:", ent.id)
        print("Image name:", ent.basename)
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
        print('Timestamp:', ent.timestamp)
        print('Model used:', ent.egg_counting_model_id)