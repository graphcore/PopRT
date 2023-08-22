# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import sys

import numpy as np
import torch

from roformer import (
    RoFormerConfig,
    RoFormerForSequenceClassification,
    RoFormerTokenizer,
)

import poprt

sys.path.append('../')
import helper

# Load RoformerV2
pretrained_model = 'junnyu/roformer_v2_chinese_char_small'
tokenizer = RoFormerTokenizer.from_pretrained(pretrained_model)
config = RoFormerConfig.from_pretrained(pretrained_model)
config.is_decoder = False
config.eos_token_id = tokenizer.sep_token_id
config.pooler_activation = "linear"
model = RoFormerForSequenceClassification.from_pretrained(
    pretrained_model, config=config
)

# seq_len = 128
input_data = (
    torch.ones(1, 128).long(),
    torch.ones(1, 128).long(),
    torch.ones(1, 128).long(),
)

print("Export ONNX model with dynamic batch size:")
torch.onnx.export(
    model,
    input_data,
    "./RoformerV2.onnx",
    export_params=True,
    opset_version=13,
    input_names=["input_ids", "token_type_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        'input_ids': {0: 'batch_size'},
        'token_type_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
    },
)
print("Check precision between Pytorch and ONNX:")
# Generate input data
input_data = {}
input_data.update(helper.generate_data({"input_ids": ([1, 128], np.int64)}, 0, 2000))
input_data.update(helper.generate_data({"token_type_ids": ([1, 128], np.int64)}, 0, 0))
input_data.update(helper.generate_data({"attention_mask": ([1, 128], np.int64)}, 1, 1))

# Run Torch
with torch.no_grad():
    input_ids = torch.from_numpy(input_data["input_ids"]).long()
    token_type_ids = torch.from_numpy(input_data["token_type_ids"]).long()
    attention_mask = torch.from_numpy(input_data["attention_mask"]).long()
    torch_res = model(input_ids, token_type_ids, attention_mask).logits
    torch_res = [torch_res.detach().numpy()]

# Run ONNXRUNTIME
sess = poprt.backend.get_session("RoformerV2.onnx", 1, "onnxruntime")
sess.load()
inputs_info, outputs_info = sess.get_io_info()
outputs_name = [o for o in outputs_info]
ort_res = sess.run(outputs_name, input_data)
helper.accuracy(torch_res, ort_res)
