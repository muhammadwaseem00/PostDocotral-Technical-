"""
Model loader - uses Config.model_name to select architecture.
See models.py for available architectures.
"""
from config import Config
from models import get_model as _get_model


def get_model():
    """Return model based on Config.model_name."""
    return _get_model(Config.model_name)
