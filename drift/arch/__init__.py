"""
Select correct arch
"""
import os
import importlib


def get_speaker_cls(arch):
    all_archs = [f.split('.')[0] for f in os.listdir(os.path.dirname(__file__)) if f != '__init__.py']
    assert arch in all_archs
    return getattr(importlib.import_module('drift.arch.' + arch), 'Speaker')


def get_listener_cls(arch):
    all_archs = [f.split('.')[0] for f in os.listdir(os.path.dirname(__file__)) if f != '__init__.py']
    assert arch in all_archs
    return getattr(importlib.import_module('drift.arch.' + arch), 'Listener')
