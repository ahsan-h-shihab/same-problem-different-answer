"""Secret loading: environment variable, else project .env, else a .secrets/ file.
Values are never printed by our scripts; only presence is ever logged.
"""
import os

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
SECRETS = os.path.join(ROOT, ".secrets")
ENV_FILE = os.path.join(ROOT, ".env")


def _load_dotenv():
    """Parse simple KEY=VALUE lines from .env (no external dependency)."""
    d = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v:
                    d[k.strip()] = v
    return d


_DOTENV = _load_dotenv()


def load_key(env_name, filename=None):
    v = os.environ.get(env_name)
    if v and v.strip():
        return v.strip()
    if env_name in _DOTENV:
        return _DOTENV[env_name]
    path = os.path.join(SECRETS, filename or (env_name.lower() + ".key"))
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            k = f.read().strip()
            if k:
                return k
    return None


def present(env_name, filename=None):
    return load_key(env_name, filename) is not None
