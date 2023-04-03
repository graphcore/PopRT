# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os

import numpy as np
import onnx

from onnx import TensorProto, checker, helper, mapping


def create_model0(opset_version=11):
    g0_dtype = TensorProto.FLOAT
    g0_add = helper.make_node("Add", ["X", "Y"], ["Z"])
    g0_reshape = helper.make_node("Reshape", ["Z", "C"], ["O"])
    g0 = helper.make_graph(
        [g0_add, g0_reshape],
        'graph',
        [
            helper.make_tensor_value_info("X", g0_dtype, (1, 2)),
            helper.make_tensor_value_info("Y", g0_dtype, (1, 2)),
        ],
        [
            helper.make_tensor_value_info("O", g0_dtype, (2,)),
        ],
    )

    g0_const_type = TensorProto.INT64
    g0_const = helper.make_tensor(
        "C",
        g0_const_type,
        (1,),
        vals=np.array([2], dtype=mapping.TENSOR_TYPE_TO_NP_TYPE[g0_const_type])
        .flatten()
        .tobytes(),
        raw=True,
    )
    g0.initializer.append(g0_const)
    m0 = helper.make_model(g0, opset_imports=[helper.make_opsetid("", opset_version)])
    checker.check_model(m0)
    return m0


def create_model1(opset_version=11):
    g1_dtype = TensorProto.FLOAT16
    g1_concat = helper.make_node("Concat", ["X", "C"], ["O"], axis=1)
    g1 = helper.make_graph(
        [g1_concat],
        'graph',
        [
            helper.make_tensor_value_info("X", g1_dtype, (1, 1)),
        ],
        [
            helper.make_tensor_value_info("O", g1_dtype, (1, 3)),
        ],
    )

    g1_const = helper.make_tensor(
        "C",
        g1_dtype,
        (1, 2),
        vals=np.array([[1.5, 2.0]], dtype=mapping.TENSOR_TYPE_TO_NP_TYPE[g1_dtype])
        .flatten()
        .tobytes(),
        raw=True,
    )
    g1.initializer.append(g1_const)
    m1 = helper.make_model(g1, opset_imports=[helper.make_opsetid("", opset_version)])
    checker.check_model(m1)
    return m1


def create_onnx(opset):
    model0 = create_model0(opset)
    model1 = create_model1(opset)

    onnx.save(model0, "model0.onnx")
    onnx.save(model1, "model1.onnx")


if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))
    if os.getcwd() != abs_path:
        raise RuntimeError(f"Please run program in {abs_path}")

    create_onnx(opset=11)

    cmd = "poprt \
            --config_yaml config.yaml "

    os.system(cmd)
