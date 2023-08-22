# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import sys

import numpy as np
import tensorflow as tf

import poprt

sys.path.append('../')
import helper

if __name__ == '__main__':
    print("Check precision between TF and ONNX:")
    model = tf.saved_model.load("open_wide_deep_saved_model")
    # Generate input data
    input_info = {
        "new_categorical_placeholder:0": ([26, 2], np.int64),
        "new_numeric_placeholder:0": ([1, 13], np.float32),
    }
    input_data = helper.generate_data(input_info, 0, 25)
    # Run TF
    tf_inputs = {
        "new_categorical_placeholder:0": tf.convert_to_tensor(
            input_data["new_categorical_placeholder:0"], tf.int64
        ),
        "new_numeric_placeholder:0": tf.convert_to_tensor(
            input_data["new_numeric_placeholder:0"], tf.float32
        ),
    }
    tf_res = model.signatures["serving_default"](**tf_inputs)
    tf_res = [tf_res["import/head/predictions/probabilities:0"].numpy()]

    # Run ONNXRUNTIME
    sess = poprt.backend.get_session("WideDeep.onnx", 1, "onnxruntime")
    sess.load()
    inputs_info, outputs_info = sess.get_io_info()
    outputs_name = [o for o in outputs_info]
    ort_inputs = {}
    ort_inputs["new_categorical_placeholder_0"] = input_data[
        "new_categorical_placeholder:0"
    ]
    ort_inputs["new_numeric_placeholder_0"] = input_data["new_numeric_placeholder:0"]
    ort_res = sess.run(outputs_name, ort_inputs)
    helper.accuracy(tf_res, ort_res)
