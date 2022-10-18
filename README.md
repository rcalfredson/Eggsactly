# counting-3
Yang Lab code related to egg counting

## Starting a local instance of the egg counting server
The server relies on a separate process, the GPU worker, which handles GPU-dependent image processing and returns the results. These processes can run in the same or different environments, depending on your setup and needs. Further, for high-capacity use cases, multiple GPU workers can support a single server.
- Install dependencies: `pip install -r requirements.txt`
  - Alternatively, use your preferred package-management tool, e.g., `virtualenv`
- Set up environment variables in `.env` file
  - For server and GPU worker instances
    - `EGG_COUNTING_BACKEND_TYPE`- how to store data used by the egg-counting server
      - `sql`- the only option currently supported.
      - `filesystem`- incompletely implemented; intended to rely solely on files stored on the server.
    - `SQL_ADDR_TYPE`- if using the SQL backend type, which format to use for the connection.
      - `sqlite`- use a DB stored on the server as a self-contained file (recommended for quick setup).
      - `shortname`- use a Unix socket to connect to a Google Cloud MySQL instance. `unix_socket` query parameter that follows `/cloudsql/` in a Google Cloud MySQL instance, e.g., `/cloudsql/${a_shortname}`.
      - `ip_addr`- use a private IP address to connect to a Google Cloud MySQL instance (an alternative to using a Unix socket). refers to a private IP address associated with a Google Cloud MySQL instance (an alternative to `shortname`, but used to create the same connection).
    - `SECRET_KEY`- required by Flask for session management; namely, it's used when encrypting cookies.
    - `NUM_GPU_WORKERS`- the number of GPU workers supporting the egg-counting server, which is used to determine how many public keys should attempt to be loaded (see description of `PRIVATE_KEY_PATH` in the following section).
    - `GPU_WORKER_TIMEOUT`- maximum duration in seconds that the egg-counting server waits before responding to a request from a GPU worker. The server responds immediately if a task becomes available before this timeout.
    - `GOOGLE_SQL_CONN_NAME`- the portion of the `unix_socket` query parameter that follows `/cloudsql/`, e.g., `/cloudsql/${GOOGLE_SQL_CONN_NAME}`; this is required if using `shortname` for `SQL_ADDR_TYPE`.
    - `GOOGLE_SQL_DB_DVT_IP`- the IP address of the Google Cloud MySQL instance; this is required if using `ip_addr` for `SQL_ADDR_TYPE`.
    - `GOOGLE_SQL_DB_PASSWORD`- required if using a Google Cloud MySQL instance (applies to both Unix socket or IP address-based connections).
    - `GOOGLE_CLIENT_ID`- client ID for a Google API application corresponding to your egg-counting server. Required only if you would like to offer Google-based OAuth 2 authentication as a method to define user accounts.
    - `GOOGLE_CLIENT_SECRET`- client secret for the above-mentioned Google Cloud application. Must be given together with `GOOGLE_CLIENT_ID`.
  - For GPU worker instances only
    - `MAIN_SERVER_URI`- URI of the egg-counting server instance that the GPU worker will support.
    - `PRIVATE_KEY_PATH`- path to the public key, in `.pem` format, that the GPU worker uses to generate tokens to access the egg-counting server. Public keys should be stored in `project/auth` using the naming convention `gpu_worker_{int}_id_rsa.pub` where `int` must range from 1 to the total number of keys (e.g., 1, 2, 3, 4). To authenticate a GPU worker request, the server tries to decode the token using these public keys.
    - `GPU_WORKER_RECONNECT_ATTEMPT_DELAY`- duration in seconds for the GPU worker to wait after a failed connection to the egg-counting server before trying again.
- Establish a SQL instance and configure its databases. Two instance types are currently supported: MySQL from Google Cloud and SQLite (DB stored on the server as a self-contained file). Google Cloud setup must be performed in advance (for more information, reference their docs). However, if using SQLite, the commands below both create and configure the database. From the Python command line:
```py
>>> from project import db, create_app
>>> with create_app().app_context():
>>>   db.create_all()
```
- Start the server
```bash
python -m project.server # optional: --host 0.0.0.0 --port 5000
# in a separate tab or different environment
python -m project.gpu_backend.worker
```


