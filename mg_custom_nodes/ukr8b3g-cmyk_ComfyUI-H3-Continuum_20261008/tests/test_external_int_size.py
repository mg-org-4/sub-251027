"""Issue #25 leaves backend dimensions, execution and identity unchanged."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ComfyUI_H3_Continuum_Join.v3.driving_nodes import H3ContinuumSamplerV38, H3ContinuumSamplerV39
from ComfyUI_H3_Continuum_Join.v3.resolution import resolve_h3_size_source

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('cls', [H3ContinuumSamplerV38, H3ContinuumSamplerV39])
def test_external_size_schema_remains_native_required_int(cls):
    required = cls.INPUT_TYPES()['required']
    assert list(required).index('height') == list(required).index('width') + 1
    for axis in ('width', 'height'):
        kind, options = required[axis]
        assert kind == 'INT'
        assert options['default'] == (736 if axis == 'width' else 416)
        assert (options['min'], options['max'], options['step']) == (32, 16384, 32)
        assert 'forceInput' not in options and 'defaultInput' not in options


@pytest.mark.parametrize('width,height', [(641,768),(640,767),(0,768),(True,768),(640,False),(16416,768)])
def test_external_resolved_dimension_uses_existing_validation(width, height):
    with pytest.raises((TypeError, ValueError)):
        resolve_h3_size_source(size_source='Manual', width=width, height=height)


def test_external_resolved_dimension_equals_manual_and_missing_first_fallback():
    manual = resolve_h3_size_source(size_source='Manual', width=640, height=768)
    assert (manual.width, manual.height) == (640,768)
    fallback = resolve_h3_size_source(size_source='First Image', width=640, height=768)
    assert (fallback.width, fallback.height) == (manual.width, manual.height)


def test_external_dimensions_do_not_override_active_first_image():
    class Image:
        shape = (1,1080,1920,3)
    expected = resolve_h3_size_source(size_source='First Image', first_frame=Image())
    external = resolve_h3_size_source(size_source='First Image', width=641, height=True, first_frame=Image())
    assert external == expected


def test_external_int_frontend_gate():
    runner = shutil.which('node')
    assert runner, 'Node.js is required for frontend tests'
    result = subprocess.run([runner, str(ROOT/'tests/frontend_external_int_size.cjs')],
                            env={**os.environ, 'H3_TEST_ROOT':str(ROOT)},
                            capture_output=True, text=True, encoding='utf-8', timeout=45)
    records = json.loads(result.stdout)
    assert len(records) == 22
    failures = [r for r in records if not r['pass']]
    assert result.returncode == 0 and not failures, result.stdout + result.stderr
