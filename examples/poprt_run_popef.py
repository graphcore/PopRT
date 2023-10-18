# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os
import tempfile

import numpy as np

from host_concat_runner_example import make_model
from lightrunner_example import default_model
from onnx import TensorProto, checker, helper

from poprt.compiler import Compiler, CompilerOptions
from poprt.converter import MFConverter
from poprt.utils import onnx_tensor_dtype_to_np_dtype


def run_popef(cmd):
    os.system(cmd)


def compile(model, file_path, model_fusion=False):
    outputs = [o.name for o in model.graph.output] if not model_fusion else None
    if model_fusion:
        outputs = None
    options = CompilerOptions()
    Compiler.compile_and_export(model, outputs, file_path, options)


def gen_model_fusion_model():
    def _create_model_0(opset_version=11):
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
            vals=np.array([2], dtype=onnx_tensor_dtype_to_np_dtype(g0_const_type))
            .flatten()
            .tobytes(),
            raw=True,
        )
        g0.initializer.append(g0_const)
        m0 = helper.make_model(
            g0, opset_imports=[helper.make_opsetid("", opset_version)]
        )
        checker.check_model(m0)
        return m0

    def _create_model_1(opset_version=11):
        g1_dtype = TensorProto.FLOAT16
        g1_mul = helper.make_node("Mul", ["X", "C"], ["Y"])
        g1_concat = helper.make_node("Concat", ["Y", "C"], ["O"], axis=1)
        g1 = helper.make_graph(
            [g1_mul, g1_concat],
            'graph',
            [
                helper.make_tensor_value_info("X", g1_dtype, (2, 1)),
            ],
            [
                helper.make_tensor_value_info("O", g1_dtype, (2, 2)),
            ],
        )

        g1_const = helper.make_tensor(
            "C",
            g1_dtype,
            (2, 1),
            vals=np.array([[1.5], [2.0]], dtype=onnx_tensor_dtype_to_np_dtype(g1_dtype))
            .flatten()
            .tobytes(),
            raw=True,
        )
        g1.initializer.append(g1_const)
        m1 = helper.make_model(
            g1, opset_imports=[helper.make_opsetid("", opset_version)]
        )
        checker.check_model(m1)
        return m1

    def _create_model_2(opset_version=11):
        g2_dtype = TensorProto.INT8
        g2_sub = helper.make_node("Sub", ["X", "C"], ["O"])
        g2_add = helper.make_node("Add", ["X", "C"], ["O2"])
        g2 = helper.make_graph(
            [g2_sub, g2_add],
            'graph',
            [
                helper.make_tensor_value_info("X", g2_dtype, (1,)),
            ],
            [
                helper.make_tensor_value_info("O", g2_dtype, (1,)),
                helper.make_tensor_value_info("O2", g2_dtype, (1,)),
            ],
        )

        g2_const = helper.make_tensor(
            "C",
            g2_dtype,
            (1,),
            vals=np.array([3], dtype=onnx_tensor_dtype_to_np_dtype(g2_dtype))
            .flatten()
            .tobytes(),
            raw=True,
        )
        g2.initializer.append(g2_const)
        m2 = helper.make_model(g2, opset_imports=[helper.make_opsetid("", 11)])
        checker.check_model(m2)
        return m2

    # create models
    models = [_create_model_0(), _create_model_1(), _create_model_2()]

    # test converter & pass
    converter = MFConverter()
    model = converter.convert(models)
    return model


def gen_pack_runner_model():
    data_shape = [3, 10]

    inputs = [
        helper.make_tensor_value_info("attention_mask", TensorProto.FLOAT, data_shape),
        helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, data_shape),
    ]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT16, data_shape)]
    nodes = [helper.make_node("Mul", ["input_ids", "attention_mask"], ["output"])]
    graph = helper.make_graph(nodes, "test_pack", inputs, outputs)
    opset_imports = [helper.make_opsetid("", 11)]
    model = helper.make_model(graph, opset_imports=opset_imports)
    return model


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        # lightRunner
        proto = default_model()
        file_path = tmpdir + "/" + "light_runner.popef"
        compile(proto, file_path)
        cmd = f"poprt_exec poprt_run_popef --popef {file_path} --runner lightRunner"
        run_popef(cmd)

        # hostConcatRunner
        proto = make_model()
        file_path = tmpdir + "/" + "host_concat_runner.popef"
        compile(proto, file_path)
        cmd = (
            f"poprt_exec poprt_run_popef --popef {file_path} --runner hostConcatRunner"
        )
        run_popef(cmd)

        # modelRunner
        cmd = f"poprt_exec poprt_run_popef --popef {file_path}"
        run_popef(cmd)

        # modelFusionRunner
        proto = gen_model_fusion_model()
        file_path = tmpdir + "/" + "model_fusion_runner.popef"
        compile(proto, file_path, model_fusion=True)
        cmd = f"poprt_exec poprt_run_popef --popef {file_path} --iteration 1000 --numThreads 1 --index 0"
        run_popef(cmd)

        # packRunner
        proto = gen_pack_runner_model()
        file_path = tmpdir + "/" + "pack_runner.popef"
        compile(proto, file_path)
        cmd = f"poprt_exec poprt_run_popef --popef {file_path} --runner packRunner --numThreads 12 --maxValidNum 10 --packTimeout 2000 --dynamicInputName input_ids --maskName attention_mask  --batchSize 1"
        run_popef(cmd)
