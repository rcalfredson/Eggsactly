# adapted from this blog post:
# https://blog.miguelgrinberg.com/post/flask-socketio-and-the-user-session

import functools
from flask_login import current_user
from flask_socketio import disconnect


def authenticated_only(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            return f({'no_auth': True})
        else:
            args[0]['no_auth'] = False
            return f(*args, **kwargs)

    return wrapped
