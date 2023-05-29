# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse

import onnx

from poprt import Pass
from poprt.passes.onnx_helper import clean_info, get_dtype, topological_sort
from poprt.passes.pattern_helper import PatternMatcher
from poprt.passes.shape_inference import infer_shapes


class PackedDeberta(Pass):
    @staticmethod
    def _find(items, search_func, return_all=False):
        results = []
        for i, item in enumerate(items):
            if search_func(item):
                results.append((i, item))
                if not return_all:
                    break
        return results if return_all else (-1, None) if not results else results[0]

    def __init__(self):
        super().__init__()

    def _add_unpack(self, model):
        max_valid_num, segment_max_size, segment_num = 2 * 8, 128, 1

        pattern = [
            's:0->Reshape:Reshape->Gather:Gather->e:3',
        ]
        pattern_matcher = PatternMatcher(pattern)
        ops = pattern_matcher.next_pattern(model.graph)
        if ops:
            unpack_info = onnx.helper.make_tensor_value_info(
                'unpack_info',
                onnx.TensorProto.INT32,
                (max_valid_num, segment_num),
            )
            model.graph.input.append(unpack_info)

            unpack_attributes = {
                'max_valid_num': max_valid_num,
                'segment_max_size': [segment_max_size],
            }
            Unpack = onnx.helper.make_node(
                'Unpack',
                name='Unpack',
                inputs=[ops['Reshape'].node.output[0], unpack_info.name],
                outputs=['{}_Unpack:0'.format(ops['Reshape'].node.output[0])],
                domain='ai.graphcore',
                **unpack_attributes,
            )
            ops['Gather'].node.input[0] = Unpack.output[0]
            model.graph.node.insert(ops['Gather'].index, Unpack)

        return model

    def _modify_mask_before_conv_and_mul(self, model):
        # conv
        pattern = [
            's:0->Sub:Sub->Cast:Cast->Unsqueeze:Unsqueeze->Expand:Expand->e:5',
        ]
        pattern_matcher = PatternMatcher(pattern)
        ops = pattern_matcher.next_pattern(model.graph)
        if ops:
            # to:
            # attention_mask->Cast->Unsqueeze->Not->(Conv)
            #                                ->Cast->(Mul)
            template = 'New_Mask'
            Cast = onnx.helper.make_node(
                'Cast',
                name='{}_Cast'.format(template),
                inputs=[ops['Sub'].node.input[1]],
                outputs=['{}_Cast:0'.format(template)],
                to=onnx.TensorProto.BOOL,
            )
            ops['Unsqueeze'].node.input[0] = Cast.output[0]
            Not = onnx.helper.make_node(
                'Not',
                name='{}_Not'.format(template),
                inputs=[ops['Unsqueeze'].node.output[0]],
                outputs=['{}_Not'.format(template)],
            )
            ops['Expand'].node.input[0] = Not.output[0]
            Cast2 = onnx.helper.make_node(
                'Cast',
                name='{}_Cast2'.format(template),
                inputs=[ops['Unsqueeze'].node.output[0]],
                outputs=['{}_Cast2:0'.format(template)],
                to=onnx.TensorProto.FLOAT16,
            )
            # add nodes
            model.graph.node.insert(ops['Unsqueeze'].index + 1, Cast2)
            model.graph.node.insert(ops['Unsqueeze'].index + 1, Not)
            model.graph.node.insert(ops['Unsqueeze'].index, Cast)
            # delete nodes
            model.graph.node.remove(ops['Sub'].node)
            model.graph.node.remove(ops['Cast'].node)
        else:
            return model
        # mul
        pattern = [
            's:0->Unsqueeze:Unsqueeze->Cast:Cast->Mul:Mul->Transpose:Transpose->e:4',
            'Cast:Cast->Mul:Mul6->Add:Add->Reshape:Reshape->e:7',
        ]
        pattern_matcher = PatternMatcher(pattern)
        ops = pattern_matcher.next_pattern(model.graph)
        mul_lst = []
        while ops:
            mul_lst.append(ops['Mul'].node)
            mul_lst.append(ops['Mul6'].node)

            model.graph.node.remove(ops['Unsqueeze'].node)
            model.graph.node.remove(ops['Cast'].node)
            ops = pattern_matcher.next_pattern(model.graph)
        for Mul in mul_lst:
            Mul.input[1] = Cast2.output[0]
        return model

    def _modify_mask_between_softmax(self, model):
        # del useless nodes
        pattern = [
            's:0->Reshape:Reshape->Squeeze:2->Unsqueeze:3->Cast:4->Mul:5->e:6',
            'Reshape:Reshape->Cast:7                     ->Mul:5',
        ]
        pattern_matcher = PatternMatcher(pattern)
        ops = pattern_matcher.next_pattern(model.graph)
        if ops:
            input = ops['Reshape'].node.input[0]
            for node in [ops[key].node for key in ['Reshape', '2', '3', '4', '5', '7']]:
                model.graph.node.remove(node)
        else:
            return model
        # add custom AttentionMask
        pattern = [
            's:0->Cast:Cast0->Sub:1->Cast:2->WhereV2:3->Reshape:Reshape4->e:5',
            'Sub:1->Cast:6->WhereV2:7->Softmax:Softmax->WhereV2:3',
            's:9->Reshape:Reshape   ->WhereV2:7',
        ]
        pattern_matcher = PatternMatcher(pattern)
        ops = pattern_matcher.next_pattern(model.graph)
        AttentionMask, AttentionMaskNot = None, None
        while ops:
            if AttentionMask is None:
                dtype = get_dtype(model.graph, ops['Reshape'].node.input[0])
                kwargs = {
                    "dataType": 'FLOAT'
                    if dtype == onnx.TensorProto.FLOAT
                    else 'FLOAT16'
                }
                AttentionMask = onnx.helper.make_node(
                    'AttentionMask',
                    name='AttentionMask',
                    inputs=[input, ops['Reshape'].node.output[0]],
                    outputs=['{}_AttentionMask'.format(ops['Reshape'].node.output[0])],
                    domain='ai.graphcore',
                    **kwargs,
                )
                Cast = onnx.helper.make_node(
                    'Cast',
                    name='{}_Cast'.format(AttentionMask.name),
                    inputs=[AttentionMask.output[0]],
                    outputs=['{}_Cast:0'.format(AttentionMask.name)],
                    to=onnx.TensorProto.BOOL,
                )
                Not = onnx.helper.make_node(
                    'Not',
                    name='{}_Not'.format(Cast.name),
                    inputs=[Cast.output[0]],
                    outputs=['{}_Not:0'.format(Cast.name)],
                )
                AttentionMaskNot = onnx.helper.make_node(
                    'Cast',
                    name='{}_Cast'.format(Not.name),
                    inputs=[Not.output[0]],
                    outputs=['{}_Cast:0'.format(Not.name)],
                    to=onnx.TensorProto.FLOAT16,
                )
                model.graph.node.insert(ops['Softmax'].index, AttentionMaskNot)
                model.graph.node.insert(ops['Softmax'].index, Not)
                model.graph.node.insert(ops['Softmax'].index, Cast)
                model.graph.node.insert(ops['Softmax'].index, AttentionMask)
            # before softmax
            Add = onnx.helper.make_node(
                'Add',
                name='{}_Add'.format(ops['Reshape'].node.output[0]),
                inputs=[AttentionMask.output[0], ops['Reshape'].node.output[0]],
                outputs=['{}_Add:0'.format(ops['Reshape'].node.output[0])],
            )
            ops['Softmax'].node.input[0] = Add.output[0]
            Mul = onnx.helper.make_node(
                'Mul',
                name='{}_Mul'.format(ops['Softmax'].node.output[0]),
                inputs=[AttentionMaskNot.output[0], ops['Softmax'].node.output[0]],
                outputs=['{}_Mul'.format(ops['Softmax'].node.output[0])],
            )
            ops['Reshape4'].node.input[0] = Mul.output[0]
            softmax_index, _ = self._find(
                model.graph.node, lambda n: n.name == ops['Softmax'].node.name
            )
            model.graph.node.insert(softmax_index + 1, Mul)
            model.graph.node.insert(softmax_index, Add)
            # del useless nodes
            for key in ('Cast0', '1', '2', '3', '6', '7'):
                model.graph.node.remove(ops[key].node)

            ops = pattern_matcher.next_pattern(model.graph)

        return model

    def packed_deberta(self, model):
        model = self._add_unpack(model)
        model = self._modify_mask_before_conv_and_mul(model)
        model = self._modify_mask_between_softmax(model)
        sorted_nodes = topological_sort(model.graph)
        model.graph.ClearField('node')
        for node in sorted_nodes:
            model.graph.node.append(node)
        return model

    def __call__(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model = self.packed_deberta(model)
        model = infer_shapes(clean_info(model))
        return model


parser = argparse.ArgumentParser(description='deberta modifier')
parser.add_argument('input', type=str, help='input model path')
args = parser.parse_args()
pack_deberta_modifer = PackedDeberta()
model = onnx.load(args.input)
model = pack_deberta_modifer(model)
output_name = "packed_" + args.input
onnx.save(model, output_name)
print(f"modified model, name: {output_name}")
