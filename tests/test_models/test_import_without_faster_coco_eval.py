# Copyright (c) OpenMMLab. All rights reserved.
"""Verify that mmdet.models can be imported without faster_coco_eval.

faster_coco_eval is an optional dependency. Importing models must not
transitively pull in mmdet.evaluation, which is the only module that requires
faster_coco_eval at import time.
"""
import subprocess
import sys
import unittest


class TestImportModelsWithoutFasterCocoEval(unittest.TestCase):

    def test_import_models_without_faster_coco_eval(self):
        """mmdet.models must be importable when faster_coco_eval is absent."""
        code = (
            'import sys\n'
            # Block faster_coco_eval so any attempt to import it raises
            # ImportError, simulating an environment where it is not installed.
            "sys.modules['faster_coco_eval'] = None\n"
            'import mmdet.models\n')
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=('Importing mmdet.models failed when faster_coco_eval is not '
                 f'installed.\nstderr:\n{result.stderr}'),
        )
