from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import datetime
import jwt
from jwt.exceptions import InvalidTokenError


class AuthDecoder:
    def __init__(self, keypaths):
        self.public_keys = []
        for keypath in keypaths:
            with open(keypath, "rb") as f:
                self.public_keys.append(
                    serialization.load_pem_public_key(
                        f.read(), backend=default_backend()
                    )
                )

    def decode_token(self, token):
        for k in self.public_keys:
            try:
                return jwt.decode(
                    token,
                    k,
                    algorithms=["RS256"],
                )
            except InvalidTokenError as exc:
                print("error while decoding the token:", exc)
                print(type(exc))
        raise InvalidTokenError
