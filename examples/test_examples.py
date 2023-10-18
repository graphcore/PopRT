# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os.path as _osp
import pathlib
import subprocess
import sys

import pytest

import poprt

HERE = pathlib.Path(__file__).parent.resolve()
PY = sys.executable

ipu21_only = pytest.mark.skipif(
    poprt.runtime.DeviceManager().ipu_hardware_version() != 'ipu21',
    reason='Skip for non-C600 IPUs.',
)


def _run_cmd(cmd: str, cwd=None):
    p = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True,
        executable='/bin/bash',
        cwd=cwd,
    )
    p.communicate()
    if p.returncode != 0:
        raise Exception(f'CMD {cmd} failed')
    else:
        return 0


JUST_RUN_PYS = []
# TDOO: enable this test
# JUST_RUN_PYS.append('convert_compile_and_run.py')
JUST_RUN_PYS.append('dynamic_batch_size.py')
JUST_RUN_PYS.append('flops_of_model.py')
JUST_RUN_PYS.append('simple_overlapio.py')
JUST_RUN_PYS.append('print_tile_mapping.py')
# TDOO: enable this test
# JUST_RUN_PYS.append('lightruner_example.py')


@pytest.mark.parametrize("just_run_py", JUST_RUN_PYS)
def test_just_run_py(just_run_py: str):
    cmd = f'{PY} {just_run_py}'
    _run_cmd(cmd, cwd=HERE)


@ipu21_only
def test_model_fusion():
    cwd = _osp.join(HERE, 'model_fusion')
    cmd = f'{PY} compile.py'
    _run_cmd(cmd, cwd=cwd)
    cmd = f'{PY} run.py'
    _run_cmd(cmd, cwd=cwd)


def test_custom_transform_example():
    cwd = _osp.join(HERE, 'custom_transform_example')
    cmd = f"bash build.sh"
    _run_cmd(cmd, cwd=cwd)
    cmd = f"{PY} test_custom_transform.py"
    _run_cmd(cmd, cwd=cwd)


def test_custom_pattern_example():
    cwd = _osp.join(HERE, 'custom_pattern_example')
    cmd = f"bash build.sh"
    _run_cmd(cmd, cwd=cwd)
    cmd = f"{PY} test_custom_pattern.py"
    _run_cmd(cmd, cwd=cwd)


def test_custom_op_example():
    cwd = _osp.join(HERE, 'custom_op_example')
    cmd = f"bash build.sh"
    _run_cmd(cmd, cwd=cwd)
    cmd = f"{PY} create_onnx_with_custom_op.py"
    _run_cmd(cmd, cwd=cwd)


def test_custom_pass():
    cwd = _osp.join(HERE, 'custom_pass')
    cmd = f"bash load_custom_passes_in_cli.sh"
    _run_cmd(cmd, cwd=cwd)
    cmd = f"{PY} load_custom_passes.py"
    _run_cmd(cmd, cwd=cwd)
