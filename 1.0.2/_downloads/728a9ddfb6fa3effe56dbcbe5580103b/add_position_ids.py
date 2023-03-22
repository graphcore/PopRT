# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import copy
import os

import onnx

# Download model from huggingface
# - python -m transformers.onnx --model=csarron/bert-base-uncased-squad-v1 . --feature question-answering
# reference: https://huggingface.co/csarron/bert-base-uncased-squad-v1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Bert-Squad Model')
    parser.add_argument(
        '--input_model', type=str, default='', help='path of input model'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_model):
        parser.print_usage()
        raise FileNotFoundError(f'Unable to find model : {args.input_model}')

    model = onnx.load(args.input_model)

    # for packed bert, we need to export position_ids to model's input
    # step 1: remove unneed node
    rm_node_names = [
        'Shape_7',
        'Gather_9',
        'Add_11',
        'Unsqueeze_12',
        'Slice_14',
        'Constant_8',
        'Constant_10',
        'Constant_13',
    ]
    rm_nodes = []
    for node in model.graph.node:
        if node.name in rm_node_names:
            rm_nodes.append(node)

    assert len(rm_node_names) == len(rm_nodes)

    for node in rm_nodes:
        model.graph.node.remove(node)

    # step 2: add position_ids to model's input
    position_ids = copy.deepcopy(model.graph.input[0])
    position_ids.name = 'position_ids'
    model.graph.input.append(position_ids)

    for node in model.graph.node:
        if node.op_type == 'Gather' and node.name == 'Gather_18':
            node.input[1] = position_ids.name

    print(f'Save preprocessed model to bert_base_squad_pos.onnx')
    onnx.save(model, 'bert_base_squad_pos.onnx')
