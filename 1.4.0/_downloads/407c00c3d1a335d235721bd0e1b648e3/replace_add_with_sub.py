# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import onnx

from poprt.passes import Pass, register


@register('replace_add_with_sub')
class ReplaceAddwithSub(Pass):
    """Replace Add with Sub."""

    def __init__(self):
        super().__init__()

    # define the transform
    def run_transform(
        self,
        graph: onnx.GraphProto,
        is_main_graph: bool,
    ) -> onnx.GraphProto:
        for node in graph.node:
            if node.op_type == 'Add':
                node.op_type = 'Sub'
        return graph

    # define the run method
    def run(self, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        onnx_model.graph.CopyFrom(
            self.traverse_graph(onnx_model.graph, self.run_transform)
        )
        return onnx_model
