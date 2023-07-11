# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import logging
import os
import re

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import onnx

from onnx import numpy_helper

from poprt.backend import get_session
from poprt.utils import LazyImport, onnx_tensor_dtype_to_np_dtype

tf = LazyImport('tensorflow')

np.random.seed(2023)

loggers = {}


def get_logger(name='debugger') -> logging.Logger:
    """Get a Python logger by name.

    :param name: The name of the logger. Defaults to 'debugger'.

    :return: The logger instance.
    """
    if name in loggers:
        return loggers[name]

    logger = logging.getLogger('debugger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    loggers[name] = logger
    return logger


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        input_func = {}
        for kv in values:
            k, v = kv.split("=")
            input_func[k] = v
        setattr(namespace, self.dest, input_func)


class Precision:
    """A class for computing precision metrics between two sets of results.

    :param method: The precision metric(s) to compute. Valid values are 'mse', 'rmse',
        'mae', 'mape', and 'r2'. If a list of strings is provided, all metrics in the list
        will be computed.
    """

    def __init__(self, method='mse') -> None:
        if isinstance(method, str) or isinstance(method, list):
            self.method = method
        else:
            raise TypeError(
                f"Compare method should be str or a list, but recieved {type(method)}"
            )

    def mse(self, result_lhs: np.ndarray, result_rhs: np.ndarray) -> float:
        """Compute the mean squared error between two sets of results.

        :param result_lhs: The first set of results.
        :param result_rhs: The second set of results.

        :returns: The mean squared error between the two sets of results.
        """
        return np.mean(np.square(result_lhs - result_rhs))

    def rmse(self, result_lhs: np.ndarray, result_rhs: np.ndarray) -> float:
        """Compute the root mean squared error between two sets of results.

        :param result_lhs: The first set of results.
        :param result_rhs: The second set of results.

        :returns: The root mean squared error between the two sets of results.
        """
        return np.sqrt(self.mse(result_lhs, result_rhs))

    def mae(self, result_lhs: np.ndarray, result_rhs: np.ndarray) -> float:
        """Compute the mean absolute error between two sets of results.

        :param result_lhs: The first set of results.
        :param result_rhs: The second set of results.

        :returns: The mean absolute error between the two sets of results.
        """
        return np.mean(np.abs(result_lhs - result_rhs))

    def mape(self, result_lhs: np.ndarray, result_rhs: np.ndarray) -> float:
        """Compute the mean absolute percentage error between two sets of results.

        :param result_lhs: The first set of results.
        :param result_rhs: The second set of results.

        :returns: The mean absolute percentage error between the two sets of results.
        """
        return np.mean(np.abs((result_lhs - result_rhs) / result_lhs)) * 100

    def r2(self, result_lhs: np.ndarray, result_rhs: np.ndarray) -> float:
        """Compute the R^2 coefficient of determination between two sets of results.

        :param result_lhs: The first set of results.
        :param result_rhs: The second set of results.

        :returns: The R^2 coefficient of determination between the two sets of results.
        """
        ss_res = np.sum(np.square(result_lhs - result_rhs))
        ss_tot = np.sum(np.square(result_lhs - np.mean(result_lhs)))
        return 1 - ss_res / ss_tot

    def compare_impl(
        self, method: str, result_lhs: np.ndarray, result_rhs: np.ndarray
    ) -> float:
        """Compute a single precision metric between two sets of results.

        :param method: The precision metric to compute.
        :param result_lhs: The first set of results.
        :param result_rhs: The second set of results.

        :returns: The value of the specified precision metric between the two sets of results.
        """
        method_fn = getattr(self, method, None)
        if method_fn is None:
            raise ValueError(f"Unsupported precision compare method: {method}.")
        return method_fn(result_lhs, result_rhs)

    def compare(self, result_lhs: np.ndarray, result_rhs: np.ndarray) -> dict:
        """Compute one or more precision metrics between two sets of results.

        :param result_lhs: The first set of results.
        :param result_rhs: The second set of results.

        :returns: A dictionary mapping each requested precision metric to its value.
        """
        if result_lhs.shape != result_rhs.shape:
            raise ValueError(
                f"Shape miss match {result_lhs.shape} vs {result_rhs.shape}"
            )

        if isinstance(self.method, str):
            return {self.method: self.compare_impl(self.method, result_lhs, result_rhs)}
        elif isinstance(self.method, list):
            return {
                m: self.compare_impl(m, result_lhs, result_rhs) for m in self.method
            }


class ModelPrinter:
    def __init__(
        self, print_level, name=None, tab_size=2, logger: logging.Logger = get_logger()
    ):
        level_mapping = {"model": 0, "graph": 1, "node": 3, "tensor": 4}
        if print_level not in level_mapping.keys():
            raise ValueError(
                f"{print_level} is not supported in ModelPrinter, only supports model, graph and node."
            )
        self.print_level = print_level
        self.model_info = ''
        self.tab_size = tab_size
        self.name = name
        self.logger = logger

    def format_str(self, name, indent_level):
        self.model_info += f'{indent_level * self.tab_size * " "}' + name

    def pretty_shape(self, dims):
        if dims is not None:
            return [dim.dim_value for dim in dims.dim]
        return None

    def pretty_type(self, input_type):
        return onnx_tensor_dtype_to_np_dtype(input_type)

    def print_metadata(self, model, indent_level=0):
        self.format_str('Model metadata:\n', indent_level)
        self.format_str(f'ONNX version: {onnx.__version__}\n', indent_level + 1)
        self.format_str(f'IR version: {model.ir_version}\n', indent_level + 1)
        self.format_str(f'Producer name: {model.producer_name}\n', indent_level + 1)
        self.format_str(
            f'Producer version: {model.producer_version}\n', indent_level + 1
        )
        self.format_str(f'Model version: {model.model_version}\n', indent_level + 1)

    def print_inputs(self, model, indent_level=0):
        self.format_str('Model inputs:\n', indent_level)
        for input in model.graph.input:
            self.format_str(f'Name: {input.name}\n', indent_level + 1)
            self.format_str(
                f'Shape: {self.pretty_shape(input.type.tensor_type.shape)}\n',
                indent_level + 2,
            )
            self.format_str(
                f'Data type: {self.pretty_type(input.type.tensor_type.elem_type)}\n',
                indent_level + 2,
            )

    def print_outputs(self, model, indent_level=0):
        self.format_str('Model outputs:\n', indent_level)
        for output in model.graph.output:
            self.format_str(f'Name: {output.name}\n', indent_level + 1)
            self.format_str(
                f'Shape: {self.pretty_shape(output.type.tensor_type.shape)}\n',
                indent_level + 2,
            )
            self.format_str(
                f'Data type: {self.pretty_type(output.type.tensor_type.elem_type)}\n',
                indent_level + 2,
            )

    def print_graph(self, model, indent_level=0):
        self.format_str('Model graph:\n', indent_level)
        self.print_nodes(model, list(model.graph.node), indent_level + 1)

    def print_nodes(self, model, nodes, indent_level=0):
        for node in nodes:
            self.format_str(f'Node: {node.name}\n', indent_level)
            if self.print_level in ["node", "tensor"]:
                self.format_str(
                    f'Inputs: {[i for i in node.input]}\n', indent_level + 1
                )
                if self.print_level == "tensor":
                    self.print_tensors(model, list(node.input), indent_level + 2)
                self.format_str(
                    f'Outputs: {[o for o in node.output]}\n', indent_level + 1
                )
                if self.print_level == "tensor":
                    self.print_tensors(model, list(node.output), indent_level + 2)
                self.format_str(f'Op type: {node.op_type}\n', indent_level + 1)
                self.format_str('Attributes:\n', indent_level + 1)
                for attr in node.attribute:
                    self.print_attr(attr, indent_level + 2)

    def print_weight(self, weight, indent_level=0):
        self.format_str('Weight:\n', indent_level)
        weight_array = numpy_helper.to_array(weight)
        self.format_str(f'Name: {weight.name}\n', indent_level + 1)
        self.format_str(f'Shape: {weight.dims}\n', indent_level + 2)
        self.format_str(
            f'Data type: {self.pretty_type(weight.data_type)}\n', indent_level + 2
        )
        if self.name is not None and weight.name == self.name:
            self.format_str(f'Value: {weight_array}\n', indent_level + 2)

    def print_tensors(self, model, tensor_names, indent_level=0):
        for w in model.graph.initializer:
            if w.name in tensor_names:
                self.print_weight(w, indent_level)
                return

        value_infos = (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        )
        for t in value_infos:
            if t.name in tensor_names:
                self.format_str('Tensor:\n', indent_level)
                self.format_str(
                    f'Shape: {self.pretty_shape(t.type.tensor_type.shape)}\n',
                    indent_level + 2,
                )
                self.format_str(
                    f'Data type: {self.pretty_type(t.type.tensor_type.elem_type)}\n',
                    indent_level + 2,
                )
                return

    def print_attr(self, attr, indent_level=0):
        attr_proto_type_list = [
            'UNDEFINED',
            'f',
            'i',
            's',
            't',
            'g',
            'floats',
            'ints',
            'strings',
            'tensors',
            'graphs',
        ]
        # TODO: to support subgraph
        self.format_str(f'Name: {attr.name}\n', indent_level)
        self.format_str(
            f'Value: {getattr(attr, attr_proto_type_list[attr.type], None)}\n',
            indent_level,
        )

    def print_model(self, model):
        if self.name is not None:
            if self.print_level not in ["node", "tensor"]:
                # set lowest print level to print matched object.
                self.print_level = "tensor"
            nodes = [n for n in model.graph.node if re.search(self.name, n.name)]
            model_tensors = (
                list(model.graph.input)
                + list(model.graph.output)
                + list(model.graph.value_info)
                + [w for w in model.graph.initializer]
            )
            tensors = [t.name for t in model_tensors if re.search(self.name, t.name)]

            if nodes:
                self.print_nodes(model, nodes)
            elif tensors:
                self.print_tensors(model, tensors)
            else:
                raise ValueError(f"Can not find node or weight {self.name} in graph.")
        elif self.print_level == "model":
            # print meta info
            self.print_metadata(model)

            # print inputs/outputs
            self.print_inputs(model)
            self.print_outputs(model)
        else:
            # print graph
            self.print_graph(model)

        # print model info
        self.logger.info(self.model_info)


def update_weight_or_input(
    model: onnx.ModelProto, tensor_dict: dict, output_model: str
) -> None:
    """Update the weight or input tensor of an ONNX model with new data.

    :param model: The ONNX model to update.
    :param tensor_dict: A dictionary containing the new data to update the model with.
    :param output_model: The path to save the updated model to.
    """
    weight_tensor = None
    weight_tensor_list = []
    input_tensor_list = []
    for tensor_name, data_gen in tensor_dict.items():
        for tensor in model.graph.initializer:
            if tensor.name == tensor_name:
                # weight_tensor = tensor
                weight_tensor_list.append(tensor)

        for graph_input in model.graph.input:
            if graph_input.name == tensor_name:
                input_tensor_list.append(graph_input)

    missing_tensor = set(tensor_dict.keys()) - set(
        [w.name for w in weight_tensor_list] + [i.name for i in input_tensor_list]
    )
    if missing_tensor:
        raise ValueError(f"{missing_tensor} is not found in model.")

    for weight_tensor in weight_tensor_list:
        tensor_name = weight_tensor.name
        weight_info = {
            tensor_name: (
                list(weight_tensor.dims),
                onnx_tensor_dtype_to_np_dtype(weight_tensor.data_type),
            )
        }
        updated_weight_dict = get_synthetic_data(weight_info, {tensor_name: data_gen})
        weight_tensor.CopyFrom(
            numpy_helper.from_array(updated_weight_dict[tensor_name], tensor_name)
        )

    for input_tensor in input_tensor_list:
        tensor_name = input_tensor.name
        input_info = {
            tensor_name: (
                [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim],
                onnx_tensor_dtype_to_np_dtype(input_tensor.type.tensor_type.elem_type),
            )
        }
        updated_input_dict = get_synthetic_data(input_info, {tensor_name: data_gen})
        model.graph.initializer.append(
            numpy_helper.from_array(updated_input_dict[tensor_name], tensor_name)
        )

    onnx.save(model, output_model)


def sort(model: onnx.ModelProto, output_model: str) -> None:
    """Sort the nodes of an ONNX model topologically and save the sorted model.

    :param model: The ONNX model to sort.
    :param output_model: The path to save the sorted model to.
    """
    sorted_nodes = topological_sort(model.graph)
    model.graph.ClearField('node')
    for node in sorted_nodes:
        model.graph.node.append(node)
    onnx.save(model, output_model)


def get_common_tensor(
    src_model: Union[onnx.ModelProto, Any], dst_model: Union[onnx.ModelProto, Any]
) -> Set[str]:
    """Get the set of common intermediate tensors between two models.

    :param src_model: The source model.
    :param dst_model: The destination model.

    :return: The set of common intermediate tensors between the two models.
    """
    return get_all_intermediate_tensor(src_model) & get_all_intermediate_tensor(
        dst_model
    )


def get_all_intermediate_tensor(model: Union[onnx.ModelProto, Any]) -> Set[str]:
    """Get the set of all intermediate tensors in a model.

    :param model: The model to get the intermediate tensors from.

    :return: The set of all intermediate tensors in the model.
    """
    all_tensor = []
    if isinstance(model, onnx.ModelProto):
        for node in model.graph.node:
            all_tensor.extend(node.output)
    else:
        # doesn't check model is instance of tf.GraphDef to avoid import tf
        skip_ops = ['Const', 'ReadVariableOp', 'NoOp']
        for node in model.node:
            if node.op in skip_ops:
                continue
            else:
                all_tensor.append(node.name + ':0')
    return set(all_tensor)


def add_onnx_outputs(model: onnx.ModelProto, outputs: List[str]) -> None:
    """Add new output tensors to an ONNX model.

    :param model: The ONNX model to add outputs to.
    :param outputs: The names of the new output tensors to add.
    """
    model_output_names = set(o.name for o in model.graph.output)
    outputs_info = [
        onnx.ValueInfoProto(name=name) for name in set(outputs) - model_output_names
    ]
    model.graph.output.extend(outputs_info)


def get_synthetic_data(
    inputs_meta: Dict[str, Tuple[List[int], np.dtype]], input_func: Dict[str, str]
) -> Dict[str, np.ndarray]:
    """Generate synthetic data for a dictionary of input tensors.

    :param inputs_meta: A dictionary mapping input tensor names to their shapes and data types.
    :param input_func: A dictionary mapping input tensor names to the type of synthetic data to generate.

    :return: A dictionary mapping input tensor names to their generated synthetic data.
    """

    def check_shape(name, shape):
        for dim in shape:
            if not isinstance(dim, int):
                raise ValueError("f{name} contain invalid dim: {dim}.")

    feed_dicts = {}
    for meta_info in inputs_meta.items():
        shape = meta_info[1][0]
        dtype = meta_info[1][1]
        check_shape(meta_info[0], shape)

        func_name = input_func[meta_info[0]]
        if input_func.get(meta_info[0]) is None:
            raise ValueError(
                f"Input data is missing for {meta_info[0]}. Please specify the input by --input_data option."
            )
        if func_name in ['rand']:
            data = np.random.rand(*shape)
        elif func_name in ['ones', 'zeros']:
            func = getattr(np, func_name)
            data = func(shape)
        elif os.path.exists(input_func[meta_info[0]]):
            data = np.load(input_func[meta_info[0]]).reshape(shape)

        feed_dicts[meta_info[0]] = data.astype(dtype)

    return feed_dicts


def create_session_with_outputs(
    model: Union[onnx.ModelProto, Any], backend: str, dump_outputs: List[str]
) -> Any:
    """Create an inference session for an ONNX or TensorFlow model with specified output tensors.

    :param model: The model to create an inference session for.
    :param backend: The backend to use for the session.
    :param dump_outputs: The names of the output tensors to include in the session.

    :return: The created inference session.
    """
    if isinstance(model, onnx.ModelProto):
        if dump_outputs:
            add_onnx_outputs(model, dump_outputs)
        sess = get_session(model.SerializeToString(), 1, backend)
    else:
        sess = get_session(model.SerializeToString(), 1, backend)
        if dump_outputs:
            sess.set_output_names(dump_outputs)
    return sess


def check_precision(
    src_sess: Any,
    dst_sess: Any,
    input_func: Dict[str, str] = {},
    compare_method: Union[str, List[str]] = 'mse',
    name_mapping: Optional[Dict[str, str]] = None,
    dump_data: bool = False,
    dump_dir: str = 'compare_output',
    logger: logging.Logger = get_logger(),
) -> None:
    """Compare the precision of two ONNX or TensorFlow models.

    :param src_sess: The source inference session.
    :param dst_sess: The destination inference session.
    :param input_func: A dictionary mapping input tensor names to the type of synthetic data to generate. Defaults to {}.
    :param compare_method: The method or methods to use for comparing the model outputs. Defaults to 'mse'.
    :param name_mapping: A dictionary mapping input and output tensor names between the two models. Defaults to None.
    :param dump_data: Whether to dump the input and output data for each tensor. Defaults to False.
    :param dump_dir: The directory to save the dumped data to. Defaults to 'compare_output'.
    """
    src_inputs_info, src_outputs_info = src_sess.get_io_info()
    dst_inputs_info, dst_outputs_info = dst_sess.get_io_info()

    src_out_names = [o for o in src_outputs_info]
    dst_out_names = [o for o in dst_outputs_info]

    src_feeds_dict = get_synthetic_data(src_inputs_info, input_func)
    src_outs = src_sess.run(src_out_names, src_feeds_dict)

    dst_feeds_dict = {}
    for src_key in src_feeds_dict.keys():
        # popart may optimize some input away
        if src_key in dst_inputs_info.keys():
            dst_feeds_dict[src_key] = src_feeds_dict[src_key].astype(
                dst_inputs_info[src_key][1]
            )
        if name_mapping is not None and name_mapping.get(src_key) is not None:
            dst_key = name_mapping[src_key]
            if dst_key in dst_inputs_info.keys():
                dst_feeds_dict[dst_key] = src_feeds_dict[src_key].astype(
                    dst_inputs_info[dst_key][1]
                )

    dst_outs = dst_sess.run(dst_out_names, dst_feeds_dict)

    if dump_data:
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

    src_outs_dict = {name: value for name, value in zip(src_out_names, src_outs)}
    dst_outs_dict = {name: value for name, value in zip(dst_out_names, dst_outs)}
    if name_mapping:
        compare_names = {
            key: name_mapping[key]
            for key in src_out_names
            if name_mapping[key] in dst_out_names
        }
    else:
        compare_names = {key: key for key in src_out_names if key in dst_out_names}

    for src_key, dst_key in compare_names.items():
        src_out = src_outs_dict[src_key]
        dst_out = dst_outs_dict[dst_key]
        if dump_data:
            # replace '/' to '_' to avoid breaking the path join.
            underscore_src_key = src_key.replace('/', '_')
            underscore_dst_key = dst_key.replace('/', '_')
            src_data_file_path = os.path.join(dump_dir, f"{underscore_src_key}_src.npy")
            dst_data_file_path = os.path.join(dump_dir, f"{underscore_dst_key}_dst.npy")
            np.save(src_data_file_path, src_out)
            np.save(dst_data_file_path, dst_out)

        if src_out.shape != dst_out.shape:
            logger.warning(
                f"Shape miss match between {src_key} and {dst_key}, which shapes are {src_out.shape} vs {dst_out.shape}"
            )
        else:
            results = Precision(compare_method).compare(
                src_out, dst_out.astype(src_out.dtype)
            )

            logger.info(f'{src_key} - shape: {src_out.shape}')
            for method, diff in results.items():
                logger.info(f'    {method} - diff: {diff}')


def load(model_path: str, backend: str = 'onnx') -> Union[onnx.ModelProto, Any]:
    """Load an ONNX or TensorFlow model.

    :param model_path: The path to the model file.
    :param backend: The backend to use for loading the model. Defaults to 'onnx'.

    :return: The loaded model.
    """
    if backend in ['onnxruntime', 'poprt']:
        return onnx.load(model_path)
    elif backend == 'tensorflow':
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def
    else:
        raise ValueError(
            f"Unsupported backend: {backend} used. Only support onnxruntime, poprt and tensorflow."
        )


def compare_precision(args) -> None:
    """Compare the precision of two ONNX or TensorFlow models specified by command line arguments.

    :param args: The command line arguments.
    """
    src_model = load(args.src_model, backend=args.src_backend)
    dst_model = load(args.dst_model, backend=args.dst_backend)

    input_data = args.input_data
    compare_method = [o.strip() for o in args.compare_method.split(',')]
    dump_data = args.dump_data
    dump_dir = args.dump_dir
    src_backend = args.src_backend.lower()
    dst_backend = args.dst_backend.lower()
    dump_outputs = []
    name_mapping = None

    if args.mark_outputs is not None:
        common_tensor = get_common_tensor(src_model, dst_model)
        if args.mark_outputs == "all":
            dump_outputs = list(common_tensor)
        else:
            mark_outputs = [o.strip() for o in args.mark_outputs.split(',')]
            for output in mark_outputs:
                if output not in common_tensor:
                    raise ValueError(f"{output} is not exists in both model.")
                else:
                    dump_outputs.append(output)
    elif args.mark_io_mapping is not None:
        src_tensor_names = get_all_intermediate_tensor(src_model)
        dst_tensor_names = get_all_intermediate_tensor(dst_model)
        for src_name, dst_name in args.mark_io_mapping.items():
            if src_name not in src_tensor_names or dst_name not in dst_tensor_names:
                raise ValueError(f"Naming pair {src_name}={dst_name} is invalid.")
        name_mapping = args.mark_io_mapping
    else:
        # Only check if there's common output tensors for ONNX model. check_precision will
        # automatically compare common output tensor.
        if isinstance(src_model, onnx.ModelProto) and isinstance(
            dst_model, onnx.ModelProto
        ):
            common_outputs = list(
                set([o.name for o in src_model.graph.output])
                & set([o.name for o in dst_model.graph.output])
            )
            if not common_outputs:
                raise ValueError(f"no common outputs between both model.")

    src_sess = create_session_with_outputs(src_model, src_backend, dump_outputs)
    dst_sess = create_session_with_outputs(dst_model, dst_backend, dump_outputs)

    # name_mapping contains input and output name mapping
    if name_mapping is not None:
        if src_backend == "tensorflow" or dst_backend == "tensorflow":
            is_src = True if src_backend == "tensorflow" else False
            sess = src_sess if is_src else dst_sess
            tf_all_names = name_mapping.keys() if is_src else name_mapping.values()
            tf_input_names = input_func.keys() & tf_all_names
            tf_output_names = tf_all_names - input_func.keys()

            sess.set_input_names(list(tf_input_names))
            sess.set_output_names(list(tf_output_names))

    src_sess.load()
    dst_sess.load()

    check_precision(
        src_sess,
        dst_sess,
        input_data,
        compare_method,
        name_mapping,
        dump_data,
        dump_dir,
    )


if __name__ == '__main__':
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        '--src_model',
        '--model',
        type=str,
        help='Specify the source input model. In compare subcommand, use --dst_model to specify the second model for precision comparing.',
    )
    parser = argparse.ArgumentParser(description='Numerical debugging assistant')
    subparsers = parser.add_subparsers(dest='command', help='sub-command for debugger.')
    compare = subparsers.add_parser(
        'compare',
        description='compare precision between src_model and dst_model.',
        parents=[base_parser],
    )
    # compare
    compare.add_argument(
        '--src_backend',
        type=str,
        default='onnxruntime',
        help='Specify the source model type.',
    )
    compare.add_argument(
        '--dst_model',
        type=str,
        default=None,
        help='Specify the destination model.',
    )
    compare.add_argument(
        '--dst_backend',
        type=str,
        default='poprt',
        help='Specify the destination model type.',
    )
    compare.add_argument(
        '--compare_method',
        type=str,
        default='mse',
        help='Specify the comparing method. Use , to split multiple methods.',
    )
    compare.add_argument(
        '--input_data',
        type=str,
        metavar="KEY=VAL",
        nargs="+",
        action=StoreDictKeyPair,
        help='Specify input data for each input tensor. It could be a path for numpy file or a string in [zeros, ones, rand].',
    )
    compare.add_argument(
        '--mark_outputs',
        type=str,
        help='Mark the tensor names for compare, use , to split between different tensors. If "all" is specified, all possible tensors are compared.',
    )
    compare.add_argument(
        '--mark_io_mapping',
        type=str,
        metavar="KEY=VAL",
        nargs="+",
        action=StoreDictKeyPair,
        help='Specify inputs and outputs mapping to compare result between models have different naming. It is used when src_model and dst_model use different naming for same tensor.',
    )
    compare.add_argument(
        '--dump_data',
        action='store_true',
        help='Dump tensor data to local file.',
    )
    compare.add_argument(
        '--dump_dir',
        type=str,
        default='compare_output',
        help='Directory to save dumped data.',
    )

    # modify
    modify = subparsers.add_parser(
        'modify',
        description='Modify input model with specified operation.',
        parents=[base_parser],
    )
    modify.add_argument(
        '--update',
        type=str,
        metavar="KEY=VAL",
        nargs="+",
        action=StoreDictKeyPair,
        help='Update weights with key=value map where key is the weight name and value could be a path for numpy file or a string in [zeros, ones, rand].',
    )
    modify.add_argument(
        '--extract',
        type=str,
        metavar="KEY=VAL",
        nargs="+",
        action=StoreDictKeyPair,
        help='Extract subgraph to model. Use `--extract input_names=in_name output_names=out_name` to speicify the subgraph between in_name and out_name.',
    )
    modify.add_argument(
        '--sort',
        action='store_true',
        help='Topological sort the model.',
    )
    modify.add_argument(
        '--output_model',
        type=str,
        default=None,
        help='Path to save modified model',
    )

    # print
    printer = subparsers.add_parser(
        'print', description='print input src_model.', parents=[base_parser]
    )
    printer.add_argument(
        '--level',
        type=str,
        default='model',
        choices=['model', 'graph', 'node', 'tensor'],
        help='Set print level, choices are model, graph, node and tensor.',
    )
    printer.add_argument(
        '--name',
        type=str,
        default=None,
        help='Match the tensor or node with specified name to print.',
    )
    args = parser.parse_args()

    # subcommand print
    def print_command(args):
        model = onnx.load(args.src_model)
        ModelPrinter(args.level, args.name).print_model(model)

    # subcommand compare
    def compare_command(args):
        compare_precision(args)

    # subcommand modify
    def modify_command(args):
        output_model = args.output_model
        if output_model is None:
            output_model = args.src_model + ".updated.onnx"
        if args.extract:
            input_names = [i.strip() for i in args.extract['input_names'].split(',')]
            output_names = [o.strip() for o in args.extract['output_names'].split(',')]
            onnx.utils.extract_model(
                args.src_model, output_model, input_names, output_names
            )
        elif args.update:
            model = onnx.load(args.src_model)
            update_weight_or_input(model, args.update, output_model)
        elif args.sort:
            model = onnx.load(args.src_model)
            model = sort(model, output_model)

    def error_command(args):
        raise ValueError(f"Unsupported command: {args.command}.")

    switcher = {
        'print': print_command,
        'compare': compare_command,
        'modify': modify_command,
    }
    switcher.get(args.command, error_command)(args)
