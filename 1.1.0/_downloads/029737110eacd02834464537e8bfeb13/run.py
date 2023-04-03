# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os

import numpy as np
import numpy.testing as npt

from poprt.runtime import ModelRunner, RuntimeConfig

if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))
    if os.getcwd() != abs_path:
        raise RuntimeError(f"Please run program in {abs_path}")

    popef_path = f"{abs_path}/executable.popef"

    config = RuntimeConfig()
    config.validate_io_params = False
    runner = ModelRunner(popef_path, config)

    index = np.array([0], dtype=np.uint32)
    g0_X = np.ones([1, 2], dtype=np.float32)
    g0_Y = np.ones([1, 2], dtype=np.float32) * 2
    g0_O = np.zeros([2], dtype=np.float32)

    runner.execute(
        {
            "index": index,
            "graph0/X": g0_X,
            "graph0/Y": g0_Y,
        },
        {
            "graph0/O": g0_O,
        },
    )
    npt.assert_array_equal(g0_O, np.ones([2], dtype=np.float32) * 3)

    index = np.array([1], dtype=np.uint32)
    g1_X = np.zeros([1, 1], dtype=np.float16)
    g1_O = np.zeros([1, 3], dtype=np.float16)

    runner.execute(
        {
            "index": index,
            "graph1/X": g1_X,
        },
        {
            "graph1/O": g1_O,
        },
    )
    npt.assert_array_equal(g1_O, np.array([[0.0, 1.5, 2.0]], dtype=np.float16))
