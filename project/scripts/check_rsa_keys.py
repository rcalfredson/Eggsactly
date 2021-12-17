# source: https://stackoverflow.com/a/48916883
import jwt

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# Load the key we created
with open(
    "/home/tracking/counting-3/project/auth_secrets/gpu_worker_1_id_rsa.pem", "rb"
) as key_file:
    private_key = serialization.load_pem_private_key(
        key_file.read(), password=None, backend=default_backend()
    )

# The data we're trying to pass along from place to place
data = {"user_id": 1}

# Lets create the JWT token -- this is a byte array, meant to be sent as an HTTP header
jwt_token = jwt.encode(data, key=private_key, algorithm="RS256")

print(f"data {data}")
print(f"jwt_token {jwt_token}")

# Load the public key to run another test...
with open(
    "/home/tracking/counting-3/project/configs/gpu_worker_1_id_rsa.pub", "rb"
) as key_file:
    public_key = serialization.load_pem_public_key(
        key_file.read(), backend=default_backend()
    )

# This will prove that the derived public-from-private key is valid
print(
    f'decoded with public key (internal): {jwt.decode(jwt_token, private_key.public_key(), algorithms=["RS256"])}'
)
# This will prove that an external service consuming this JWT token can trust the token
# because this is the only key it will have to validate the token.
print(
    f"decoded with public key (external): {jwt.decode(jwt_token, public_key, algorithms=['RS256'])}"
)
