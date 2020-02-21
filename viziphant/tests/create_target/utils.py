import hashlib
import os


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath, md5):
    is_valid = os.path.isfile(fpath)
    is_valid &= md5 == calculate_md5(fpath)
    if not is_valid:
        raise IOError("Invalid file: invalid checksum.")
