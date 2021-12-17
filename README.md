# counting-3
Yang Lab code related to egg counting

## Starting a local instance of the egg counting server
- Install dependencies: `pip install -r requirements.txt`
  - Alternatively, use your preferred package-management tool, e.g., `virtualenv`
```bash
export FLASK_APP=project/server.py
export FLASK_ENV=development
flask run
# in a separate tab
python -m project.gpu_backend.worker
```


