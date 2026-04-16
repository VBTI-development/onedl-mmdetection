# Copyright (c) VBTI. All rights reserved.
"""Tests whether mmseg works when onedl-mmcv without ops is installed."""

import sys
import traceback
import types

import pytest


@pytest.mark.order(0)
def test_mmdet_import_without_mmcv_ops(monkeypatch):
    # Simulate mmcv.ops not being present or failing to import

    class OpsMock(types.ModuleType):

        def __getattr__(self, name):
            if name in {
                    '__file__', '__name__', '__package__', '__loader__',
                    '__spec__'
            }:
                return super().__getattribute__(name)
            raise ModuleNotFoundError('No module named "mmcv._ext"')

    ops_mock = OpsMock('mmcv.ops')
    monkeypatch.setitem(sys.modules, 'mmcv.ops', ops_mock)
    sys.modules.pop('mmcv._ext', None)

    # Try importing mmdet
    import mmdet  # noqa: F401
    mmdet.__version__

    from mmdet.utils import register_all_modules

    try:
        register_all_modules()  # should not raise import errors
    except ModuleNotFoundError as e:
        tb = traceback.format_tb(e.__traceback__)
        last_entry = tb[-2] if tb else 'No traceback available'
        assert False, (
            'Import failed, onedl-mmcv.ops is not properly guarded in\n'
            f"{last_entry}")
