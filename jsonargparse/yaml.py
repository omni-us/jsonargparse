import copy
import os

from typing import Dict, List
from .loaders_dumpers import load_value
from .util import Path, change_to_path_dir, get_config_read_mode


def deep_update(source, override):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    if isinstance(source, Dict) and isinstance(override, Dict):
        if '__delete__' in override:
            delete_keys = override.pop('__delete__')
            if isinstance(delete_keys, str):
                delete_keys = [delete_keys]

            if isinstance(delete_keys, list):
                for k in delete_keys:
                    if k in source:
                        source.pop(k)
            elif delete_keys:
                return override
        for key, value in override.items():
            if isinstance(value, Dict) and key in source:
                source[key] = deep_update(source[key], value)
            else:
                source[key] = override[key]
        return source
    elif isinstance(source, List) and isinstance(override, Dict):
        if '__delete__' in override and override['__delete__'] is True:
            override.pop('__delete__')
            return override

        if 'change_item' in override:
            change_item = override.pop('change_item')
            for index, v in change_item:
                source[index] = deep_update(source[index], v)

        if 'insert_item' in override:
            insert_item = override.pop('insert_item')
            insert_item.sort(key = lambda x: x[0], reverse = True)
            for item in insert_item:
                if len(item) == 3:
                    index, value, extend = item
                else:
                    index, value = item
                    extend = False
                if extend:
                    assert isinstance(value, list), 'Cannot extend a non-list'
                    value.reverse()
                    for v in value:
                        source.insert(index, v)
                else:
                    source.insert(index, value)

                if '__delete__' in override:
                    if isinstance(override['__delete__'], int):
                        override['__delete__'] = [override['__delete__']]
                    for i in range(len(override['__delete__'])):
                        if override['__delete__'][i] >= index:
                            if extend:
                                override['__delete__'][i] += len(value)
                            else:
                                override['__delete__'][i] += 1

        if '__delete__' in override:
            delete_keys = override.pop('__delete__')
            if isinstance(delete_keys, int):
                delete_keys = [delete_keys]

            if isinstance(delete_keys, list):
                delete_keys = list({int(d) for d in delete_keys})
                delete_keys.sort(reverse = True)
                for k in delete_keys:
                    source.pop(k)
            elif delete_keys:
                return override
        if 'pre_item' in override:
            source = (override['pre_item'] if isinstance(override['pre_item'], list) else [override['pre_item']]) + source
        if 'post_item' in override:
            source = source + (override['post_item'] if isinstance(override['post_item'], list) else [override['post_item']])
        return source
    return override


def get_cfg_from_path(cfg_path):
    fpath = Path(cfg_path, mode = get_config_read_mode())
    with change_to_path_dir(fpath):
        cfg_str = fpath.get_content()
        parsed_cfg = load_value(cfg_str)
    return parsed_cfg


def parse_config(cfg_file, cfg_path = None, **kwargs):
    if '__base__' in cfg_file:
        sub_cfg_paths = cfg_file.pop('__base__')
        if sub_cfg_paths is not None:
            if not isinstance(sub_cfg_paths, list):
                sub_cfg_paths = [sub_cfg_paths]
            sub_cfg_paths = [sub_cfg_path if isinstance(sub_cfg_path, list) else [sub_cfg_path, ''] for sub_cfg_path in sub_cfg_paths]
            if cfg_path is not None:
                sub_cfg_paths = [[os.path.normpath(os.path.join(os.path.dirname(cfg_path), sub_cfg_path[0])) if not os.path.isabs(
                    sub_cfg_path[0]) else sub_cfg_path[0], sub_cfg_path[1]] for sub_cfg_path in sub_cfg_paths]
            sub_cfg_file = {}
            for sub_cfg_path in sub_cfg_paths:
                cur_cfg_file = parse_path(sub_cfg_path[0], **kwargs)
                for key in sub_cfg_path[1].split('.'):
                    if key:
                        cur_cfg_file = cur_cfg_file[key]
                sub_cfg_file = deep_update(sub_cfg_file, cur_cfg_file)
            cfg_file = deep_update(sub_cfg_file, cfg_file)
    if '__import__' in cfg_file:
        cfg_file.pop('__import__')

    for k, v in cfg_file.items():
        if isinstance(v, dict):
            cfg_file[k] = parse_config(v, cfg_path, **kwargs)
    return cfg_file


def parse_path(cfg_path, seen_cfg = None, **kwargs):
    abs_cfg_path = os.path.abspath(cfg_path)
    if seen_cfg is None:
        seen_cfg = {}
    elif abs_cfg_path in seen_cfg:
        if seen_cfg[abs_cfg_path] is None:
            raise RuntimeError('Circular reference detected in config file')
        else:
            return copy.deepcopy(seen_cfg[abs_cfg_path])

    cfg_file = get_cfg_from_path(cfg_path)
    seen_cfg[abs_cfg_path] = None
    cfg_file = parse_config(cfg_file, cfg_path = cfg_path, seen_cfg = seen_cfg, **kwargs)
    seen_cfg[abs_cfg_path] = cfg_file
    return cfg_file


def parse_str(cfg_str, cfg_path = None, seen_cfg = None, **kwargs):
    if seen_cfg is None:
        seen_cfg = {}
    cfg_file = load_value(cfg_str)
    if cfg_path is not None:
        abs_cfg_path = os.path.abspath(cfg_path)
        if abs_cfg_path in seen_cfg:
            if seen_cfg[abs_cfg_path] is None:
                raise RuntimeError('Circular reference detected in config file')
            else:
                return copy.deepcopy(seen_cfg[abs_cfg_path])
        seen_cfg[abs_cfg_path] = None
    cfg_file = parse_config(cfg_file, cfg_path = cfg_path, seen_cfg = seen_cfg, **kwargs)
    if cfg_path is not None:
        seen_cfg[abs_cfg_path] = cfg_file
    return cfg_file
