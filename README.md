# counting-3
Yang Lab code related to egg counting

## Starting a local instance of the egg counting server
- Install dependencies: `pip install -r requirements.txt`
  - Alternatively, use your preferred package-management tool, e.g., `virtualenv`
- Initialize a local instance of the SQLite server
- From the Python command line 
```py
from project import db, create_app
with create_app().app_context():
  db.create_all()
```
- Start the server
```bash
python -m project.server # --host 0.0.0.0 --port 5000
# in a separate tab
python -m project.gpu_backend.worker
```


