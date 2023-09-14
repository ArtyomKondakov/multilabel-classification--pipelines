"""Utils file."""
import importlib
from typing import Any


def load_object(obj_path: str, def_obj_path: str = '') -> Any:
    """Load object.

    Args:
        obj_path (str): path to object
        def_obj_path (str):  default path to object. Defaults to ''.

    Raises:
        AttributeError: _description_

    Returns:
        Any: _description_
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else def_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)

    try:
        return getattr(module_obj, obj_name)
    except AttributeError:
        raise AttributeError(
            'Object `{obj_name}` cannot be loaded from `{obj_path}`.'.format(
                obj_name=obj_name,
                obj_path=obj_path,
            ),
        )
