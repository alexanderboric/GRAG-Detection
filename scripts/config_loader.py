# config_loader.py
import yaml
import os

_config = None

def load_config(path: str):
    """Lädt die Config nur einmal global."""
    global _config
    if _config is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config-Datei '{path}' wurde nicht gefunden.")
        with open(path, "r") as f:
            _config = yaml.safe_load(f)

def get_config():
    """Gibt die globale Config zurück."""
    if _config is None:
        raise RuntimeError("Config not loaded yet! Bitte zuerst load_config() aufrufen.")
    return _config