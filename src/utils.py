import torch
from collections.abc import Mapping


def deep_update(source, overrides):
    """
    Actualizar un diccionario anidado o un mapeo similar. 
    Modificar "source" en su lugar.
    """
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source