import hashlib
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, PrivateFormat, NoEncryption, load_pem_private_key, load_pem_public_key
)

def generate_keypair():
    """Generates standard SECP256R1 ECDSA keypair in PEM format."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    private_pem = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption()
    )
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=Encoding.PEM,
        format=PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem.decode("utf-8"), public_pem.decode("utf-8")

def sign_data(private_key_pem_str: str, data_str: str) -> str:
    """Signs data string using the private key and returns signature in hex."""
    private_key = load_pem_private_key(private_key_pem_str.encode("utf-8"), password=None)
    signature = private_key.sign(
        data_str.encode("utf-8"),
        ec.ECDSA(hashes.SHA256())
    )
    return signature.hex()

def verify_signature(public_key_pem_str: str, signature_hex: str, data_str: str) -> bool:
    """Verifies ECDSA signature of data using public key PEM string."""
    try:
        public_key = load_pem_public_key(public_key_pem_str.encode("utf-8"))
        public_key.verify(
            bytes.fromhex(signature_hex),
            data_str.encode("utf-8"),
            ec.ECDSA(hashes.SHA256())
        )
        return True
    except Exception:
        return False

def hash_data(data_str: str) -> str:
    """Returns SHA256 hex hash of the string."""
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

def merkle_root(data_list: list) -> str:
    """Computes Merkle Root hash of a list of data items."""
    if not data_list:
        return hash_data("")
    
    current_level = [hash_data(str(item)) for item in data_list]
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                combined = current_level[i] + current_level[i + 1]
            else:
                # Duplicate last node if odd number of nodes
                combined = current_level[i] + current_level[i]
            next_level.append(hash_data(combined))
        current_level = next_level
    return current_level[0]
