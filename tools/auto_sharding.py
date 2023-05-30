# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import copy
import os
import re
import time

from collections import deque
from multiprocessing import Array, Process, Queue, current_process
from queue import Empty
from typing import Dict, List, Set, Tuple

import numpy as np
import onnx

from poprt import runtime
from poprt._argparse import PopHelpFormatter, argparse
from poprt.compiler import Compiler, CompilerOptions
from poprt.converter import Sharder
from poprt.profile import Profiler


class Subgraph:
    """Create a subgraph including several nodes."""

    def __init__(
        self,
        start_node: str,
        nodes: List[str],
        id: int,
        bytes_cost: float,
        flops_cost: int,
    ) -> None:
        """Create a subgraph including several nodes.

        :param start_node: the name of the start node of the subgraph.
        :param nodes: the list of nodes in the subgraph.
        :param id: the id of the subgraph.
        :param bytes_cost: the bytes cost of the subgraph.
        :param flops_cost: the FLOPs cost of the subgraph.
        """
        self.start_node = start_node
        self.nodes = nodes
        self.id = id
        self.bytes_cost = bytes_cost
        self.flops_cost = flops_cost
        # The parents of the current subgraph are the ids of subgraphs
        # whose start nodes are the input nodes of the current subgraph.
        self.parents = set()
        # If the current subgraph is the parent of other subgraphs,
        # the ids of the subgraphs will be stored in kids.
        self.kids = set()
        # The ids of the subgraphs which are able
        # to be travelled from the start node of the current subgraph.
        self.included_subgraphs = set()
        # The ids of the subgraphs which are parallel to the current subgraph.
        self.parallel_subgraphs = set()

    def is_parallel(self, other: "Subgraph") -> bool:
        """Check if other subgraph is parallel to the current subgraph.

        :param other: the other subgraph.
        """
        if (
            other.id not in self.included_subgraphs
            and self.id not in other.included_subgraphs
            and self.id != other.id
        ):
            return True
        else:
            return False

    def __eq__(self, other):
        return self.id == other.id


class SubgraphGroup:
    """Create a group of subgraphs.

    A groups of subgraphs will be allocated into one IPU.
    """

    def __init__(self):
        self.subgraphs = []
        # The total bytes cost of the subgraphs group.
        self.subgraph_bytes_cost = 0.0
        # The total FLOPs cost of the subgraphs group.
        self.subgraph_flops_cost = 0
        # The start subrgraphs are the subgraphs starting with sharding nodes of the group.
        self.start_subgraphs = []
        self.is_last_subgraph = False

    def add(self, subgraph: Subgraph):
        """Add the subgraph into group.

        :param subgraph: the subgraph to be added.
        """
        self.subgraphs.append(subgraph)
        self.subgraph_bytes_cost += subgraph.bytes_cost
        self.subgraph_flops_cost += subgraph.flops_cost

        # Update start subgraphs after adding the subgraph into the group.
        if not self.is_last_subgraph:
            i = 0
            is_start = True
            while i < len(self.start_subgraphs):
                if self.start_subgraphs[i].id in subgraph.included_subgraphs:
                    self.start_subgraphs.remove(self.start_subgraphs[i])
                elif subgraph.id in self.start_subgraphs[i].included_subgraphs:
                    is_start = False
                    break
                else:
                    i += 1
            # Add the subgraph
            if is_start:
                self.start_subgraphs.append(subgraph)

    def remove(self, subgraph: Subgraph):
        """Remove the subgraph into group.

        :param subgraph: the subgraph to be removed.
        """
        # Update start subgraphs after removing the subgraph from the group.
        if not self.is_last_subgraph:
            if subgraph in self.start_subgraphs:
                self.start_subgraphs.remove(subgraph)
                for id in subgraph.parents:
                    for s in self.subgraphs:
                        if s.id == id:
                            is_start = True
                            for ss in self.start_subgraphs:
                                if s.id in ss.included_subgraphs:
                                    is_start = False
                                    break
                            if is_start:
                                self.start_subgraphs.append(s)
        self.subgraphs.remove(subgraph)
        self.subgraph_bytes_cost -= subgraph.bytes_cost
        self.subgraph_flops_cost -= subgraph.flops_cost


SKIP_OP_TYPES = ["Reshape", "Slice", "Tile", "Expand"]


class AutoSharding:
    """Auto shard ONNX model into devices and pipelining stages."""

    def __init__(
        self,
        num_ipus: int,
        optimal_perf: bool = False,
        num_processes: int = 2,
    ) -> None:
        """Construct a new AutoSharding object.

        :param num_ipus: the number of ipus.
        :param optimal_perf: If True, continue traversing to find optimal performance, else stop if find a valid solution.
        :param num_processes: the number of processes for compilation.
        """
        self._num_ipus = num_ipus
        self._optimal_perf = optimal_perf
        self._num_processes = num_processes

        # Dict[node_name, List[input_node_names]]
        self._node_to_input_nodes: Dict[str, List[str]] = {}
        # Dict[node_name, NodeProto]
        self._name_to_node_proto: Dict[str, onnx.NodeProto] = {}
        # Dict[initializers_name, initializer]
        self._name_to_initializer: Dict[str, onnx.TensorProto] = {}
        # List[candidate_shard_node_name]
        self._candidates: List[str] = []
        # Set[travelled_node_name]
        self._travelled: Set[str] = set()
        # Dict[tensor_name, shape]
        self._tensor_to_shape: Dict[str, np.ndarray] = {}
        # List[Subgraph]
        self._all_subgraphs = []

        self._total_bytes_cost = 0.0

        self._total_flops_cost = 0

    def _bytes_cost_by_nodes(self, nodes: List[onnx.NodeProto]) -> float:
        """Calculate the byte-size cost by nodes in terms of nodes' initializer inputs.

        :param nodes: the list of nodes involved in calculation.
        """

        bytes_cost = 0.0
        for node in nodes:
            if node.op_type in SKIP_OP_TYPES:
                continue
            for i in node.input:
                if i in self._name_to_initializer.keys():
                    value = onnx.numpy_helper.to_array(self._name_to_initializer[i])
                    num_elems = value.size
                    byte_per_elems = value.itemsize
                    bytes_cost += num_elems * byte_per_elems
        return bytes_cost

    def _flops_cost_by_nodes(self, nodes: List[onnx.NodeProto]) -> int:
        """Calculate the FLOPs cost by nodes.

        :param nodes: the list of nodes involved in calculation.
        """
        flops_cost = 0.0
        dummy_shape = np.array([]).astype(np.int64)
        for node in nodes:
            input_shapes = []
            for input in node.input:
                input_shape = (
                    self._tensor_to_shape[input]
                    if input in self._tensor_to_shape
                    else dummy_shape
                )
                input_shapes.append(input_shape)

            output_shapes: List[np.ndarray] = []
            for output in node.output:
                output_shape = (
                    self._tensor_to_shape[output]
                    if output in self._tensor_to_shape
                    else dummy_shape
                )
                output_shapes.append(output_shape)

            target_profiler = Profiler().get_profiler(node.op_type)
            node_flops = target_profiler(node, input_shapes, output_shapes)
            flops_cost += node_flops
        return flops_cost

    def _is_candidate(self, node_proto: onnx.NodeProto) -> bool:
        """Check if the node is a candidate node.

        :param node_proto: the node to be checked.
        """

        is_candidate = False
        valid_inputs = [
            i for i in node_proto.input if i not in self._name_to_initializer.keys()
        ]
        # The node with multiple value info inputs is a candidate node.
        if len(valid_inputs) > 1:
            is_candidate = True
        return is_candidate

    def _init_info(self, graph_proto: onnx.GraphProto) -> None:
        """Init the information for auto-sharding.

        :param graph_proto: the ONNX graph.
        """
        for i in graph_proto.initializer:
            # Init _name_to_initializer
            self._name_to_initializer[i.name] = i
            # Init _tensor_to_shape
            self._tensor_to_shape[i.name] = np.array(i.dims)

        # Init _tensor_to_shape
        for t in (
            list(graph_proto.input)
            + list(graph_proto.value_info)
            + list(graph_proto.output)
        ):
            self._tensor_to_shape[t.name] = np.array(
                [d.dim_value for d in t.type.tensor_type.shape.dim]
            )

        for node_proto in graph_proto.node:
            # Init _node_to_input_nodes
            self._node_to_input_nodes[node_proto.name] = []

            for name, proto in self._name_to_node_proto.items():
                if set(proto.output) & set(node_proto.input):
                    self._node_to_input_nodes[node_proto.name].append(name)

            # Init _name_to_node_proto
            self._name_to_node_proto[node_proto.name] = node_proto

            # Init _candidates
            if self._is_candidate(node_proto):
                self._candidates.append(node_proto.name)

            self._total_bytes_cost += self._bytes_cost_by_nodes([node_proto])
            self._total_flops_cost += self._flops_cost_by_nodes([node_proto])

        # If there is no candidate, add all nodes to candidates
        if len(self._candidates) == 0:
            self._candidates = [n.name for n in graph_proto.node]
        print(f"The number of candidates: {len(self._candidates)}")

    def _traverse_subgraph(self, start_node: str) -> Set[str]:
        """Traverse the subgraph nodes from the node.

        :param start_node: the name of the start node of the subgraph.
        """
        if start_node in self._travelled:
            return None

        subgraph_nodes = set()
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            if node not in self._travelled:
                self._travelled.add(node)
                subgraph_nodes.add(node)
                queue.extend(
                    i
                    for i in self._node_to_input_nodes[node]
                    if i not in self._travelled
                )

        return subgraph_nodes

    def _raw_sharding(self) -> List[SubgraphGroup]:
        """Shard the subgraphs into SubgraphGroups to balance bytes cost on devices."""

        avg_bytes_cost = self._total_bytes_cost / self._num_ipus

        # Find the raw shard indexes in the total subgraphs
        cum_bytes_cost = 0.0
        raw_shard_indexes = []
        for i in range(len(self._all_subgraphs)):
            s = self._all_subgraphs[i]
            if cum_bytes_cost + s.bytes_cost >= avg_bytes_cost:
                cum_bytes_cost = 0.0
                raw_shard_indexes.append(i)
                if len(raw_shard_indexes) == self._num_ipus - 1:
                    break
            else:
                cum_bytes_cost += s.bytes_cost

        # Split the subgraph_group list into SubgraphGroup list
        subgraph_groups = []
        for i in range(len(raw_shard_indexes)):
            sg = SubgraphGroup()
            last_index = 0 if i == 0 else raw_shard_indexes[i - 1] + 1
            for j in range(last_index, raw_shard_indexes[i] + 1):
                sg.add(self._all_subgraphs[j])
            subgraph_groups.append(sg)

        sg = SubgraphGroup()
        sg.is_last_subgraph = True
        for i in range(raw_shard_indexes[-1] + 1, len(self._all_subgraphs)):
            sg.add(self._all_subgraphs[i])
        subgraph_groups.append(sg)
        return subgraph_groups

    def _execute(self, popef: str) -> Tuple[float, float]:
        """Execute the PopEF.

        :param popef: the PopEF file name.
        """

        # Performance
        runner_config = runtime.RuntimeConfig()
        runner_config.timeout_ns = 0
        runner = runtime.ModelRunner(popef, runner_config)
        inputs = {}
        for i in runner.get_model_inputs():
            inputs[i.name] = np.ones(i.shape).astype(i.numpy_data_type())
        outputs = {}
        for o in runner.get_model_outputs():
            outputs[o.name] = np.zeros(o.shape, dtype=o.numpy_data_type())

        # Warmup
        futures = []
        for i in range(20):
            f = runner.executeAsync(inputs, outputs)
            futures.append(f)
        for i, future in enumerate(futures):
            future.wait()

        # Run
        futures = []
        sess_start = time.time()
        for i in range(200):
            f = runner.executeAsync(inputs, outputs)
            futures.append(f)
        for i, future in enumerate(futures):
            future.wait()
        sess_end = time.time()
        latency = (sess_end - sess_start) / 200 * 1000
        tput = 1 / latency * 1000
        return latency, tput

    def _remove_to_fit(
        self,
        group_idx: int,
        subgraph_groups: List[SubgraphGroup],
        balance_flops: bool = False,
    ) -> List[List[SubgraphGroup]]:
        """Try to find the next SubgraphGroup lists from the current SubgraphGroup list.

        :param group_idx: the index of group which is out of the memory.
        :param subgraph_groups: the list of SubgraphGroup.
        :param balance_flops: If True, enable to find the next SubgraphGroup lists based on FLOPs cost, else bytes cost.
        """

        def update(new_group_idx: int, s: Subgraph) -> List[SubgraphGroup]:
            # Remove Subgraph s from group_idx and add it to new_group_idx
            sg_t = copy.deepcopy(subgraph_groups)
            if balance_flops:
                # The original new group should have less FLOPs cost than the old group
                cond1 = (
                    sg_t[new_group_idx].subgraph_flops_cost
                    < sg_t[group_idx].subgraph_flops_cost
                )
                # The gap between updated new group and old group should be less than the gap between the original new group and old group
                cond2 = np.abs(
                    sg_t[new_group_idx].subgraph_flops_cost
                    - sg_t[group_idx].subgraph_flops_cost
                ) > np.abs(
                    sg_t[new_group_idx].subgraph_flops_cost
                    + s.flops_cost
                    - sg_t[group_idx].subgraph_flops_cost
                    + s.flops_cost
                )
                if cond1 and cond2:
                    sg_t[group_idx].remove(s)
                    sg_t[new_group_idx].add(s)
                    return sg_t
                else:
                    return None
            else:
                sg_t[group_idx].remove(s)
                sg_t[new_group_idx].add(s)
                return sg_t

        # Move subgraph to next group
        combination = []
        if group_idx < len(subgraph_groups) - 1:
            # The start_subgraph is in the bottom of the group,
            # and then they are able to be moved to the next group.

            for s in subgraph_groups[group_idx].start_subgraphs:
                next_group = None
                # Find the next SubgraphGroup which has the minimum subgraph_bytes_cost
                for j in range(group_idx + 1, len(subgraph_groups)):
                    for ss in subgraph_groups[j].subgraphs:
                        if s.id in ss.parents:
                            if next_group is None:
                                next_group = j
                            else:
                                if balance_flops:
                                    if (
                                        subgraph_groups[j].subgraph_flops_cost
                                        < subgraph_groups[
                                            next_group
                                        ].subgraph_flops_cost
                                    ):
                                        next_group = j
                                else:
                                    if (
                                        subgraph_groups[j].subgraph_bytes_cost
                                        < subgraph_groups[
                                            next_group
                                        ].subgraph_bytes_cost
                                    ):
                                        next_group = j
                            break
                if next_group is not None:
                    new_groups = update(next_group, s)
                    if new_groups is not None:
                        combination.append(new_groups)

                # Try parallel subgraphs
                # if parallel subgraph is start subgraph, it's able to be moved to parallel group
                parallel_group = None
                for j in range(len(subgraph_groups)):
                    if j != group_idx:
                        for ss in subgraph_groups[j].subgraphs:
                            if s.id in ss.parallel_subgraphs:
                                if parallel_group is None:
                                    parallel_group = j
                                else:
                                    if balance_flops:
                                        if (
                                            subgraph_groups[j].subgraph_flops_cost
                                            < subgraph_groups[
                                                parallel_group
                                            ].subgraph_flops_cost
                                        ):
                                            parallel_group = j
                                    else:
                                        if (
                                            subgraph_groups[j].subgraph_bytes_cost
                                            < subgraph_groups[
                                                parallel_group
                                            ].subgraph_bytes_cost
                                        ):
                                            parallel_group = j
                                break
                if parallel_group is not None:
                    new_groups = update(parallel_group, s)
                    if new_groups is not None:
                        combination.append(new_groups)

        # Move subgraph to previous group
        if group_idx > 0:
            # The end subgraphs is in the top of the group,
            # and then they are able to be moved to the previous group.
            end_subgraphs = []
            for s in subgraph_groups[group_idx].subgraphs:
                if not (
                    s.parents
                    & set([s.id for s in subgraph_groups[group_idx].subgraphs])
                ):
                    end_subgraphs.append(s)

            for s in end_subgraphs:
                pre_group = None
                # Find the pre SubgraphGroup which has the minimum subgraph_bytes_cost
                for j in range(group_idx):
                    for ss in subgraph_groups[j].subgraphs:
                        if s.id in ss.kids:
                            if pre_group is None:
                                pre_group = j
                            else:
                                if balance_flops:
                                    if (
                                        subgraph_groups[j].subgraph_flops_cost
                                        < subgraph_groups[pre_group].subgraph_flops_cost
                                    ):
                                        pre_group = j
                                else:
                                    if (
                                        subgraph_groups[j].subgraph_bytes_cost
                                        < subgraph_groups[pre_group].subgraph_bytes_cost
                                    ):
                                        pre_group = j
                            break
                if pre_group is not None:
                    new_groups = update(pre_group, s)
                    if new_groups is not None:
                        combination.append(new_groups)

                # Try parallel subgraphs
                # if parallel subgraph is end subgraph, it's able to be moved to parallel group
                parallel_group = None
                for j in range(len(subgraph_groups)):
                    if j != group_idx:
                        for ss in subgraph_groups[j].subgraphs:
                            if s.id in ss.parallel_subgraphs:
                                if parallel_group is None:
                                    parallel_group = j
                                else:
                                    if balance_flops:
                                        if (
                                            subgraph_groups[j].subgraph_flops_cost
                                            < subgraph_groups[
                                                parallel_group
                                            ].subgraph_flops_cost
                                        ):
                                            parallel_group = j
                                    else:
                                        if (
                                            subgraph_groups[j].subgraph_bytes_cost
                                            < subgraph_groups[
                                                parallel_group
                                            ].subgraph_bytes_cost
                                        ):
                                            parallel_group = j
                                break
                if parallel_group is not None:
                    new_groups = update(parallel_group, s)
                    if new_groups is not None:
                        combination.append(new_groups)

        # limit the number of subgraph groups
        if balance_flops:
            flops_stdv = []
            for c in combination:
                flops_stdv.append(np.std([g.subgraph_flops_cost for g in c]))
            sorted_idxs = sorted(range(len(flops_stdv)), key=lambda k: flops_stdv[k])
            # Only keep the minimum stdv subgraph groups
            combination = [combination[i] for i in sorted_idxs[:3]]
        return combination

    def _parallel_compile(
        self,
        queue: Queue,
        compile_results_queue: Queue,
        block_flags: Array,
        onnx_model: onnx.ModelProto,
    ) -> Tuple:
        """Try to compile the ONNX model with the sharding nodes.

        :param queue: the process queue of the list of SubgraphGroup.
        :param compile_results_queue: the process queue of the compile results.
        :param onnx_model: the ONNX model.
        """

        def compile(
            subgraph_groups: List[SubgraphGroup], popef_idx: int, perf_idx: int
        ):
            print(f"[Process {current_process().name}] Try to compile...")
            for i in range(len(subgraph_groups) - 1):
                print(
                    f"[Process {current_process().name}] Device {i} - Bytes_cost: {subgraph_groups[i].subgraph_bytes_cost}, FLOPs_cost: {subgraph_groups[i].subgraph_flops_cost}, Sharding nodes: {[s.start_node for s in subgraph_groups[i].start_subgraphs]}"
                )
            print(
                f"[Process {current_process().name}] Device {len(subgraph_groups) - 1} - Bytes_cost: {subgraph_groups[len(subgraph_groups) - 1].subgraph_bytes_cost}, FLOPs_cost: {subgraph_groups[len(subgraph_groups) - 1].subgraph_flops_cost}"
            )
            sharding_info = {}
            for i in range(len(subgraph_groups)):
                for start_subgraph in subgraph_groups[i].start_subgraphs:
                    sharding_info[start_subgraph.start_node] = i
            sharded_model = Sharder(sharding_info).run(copy.deepcopy(onnx_model))
            outputs = [o.name for o in sharded_model.graph.output]

            options = CompilerOptions()
            options.ipu_version = runtime.DeviceManager().ipu_hardware_version()
            options.num_ipus = self._num_ipus
            options.virtual_graph_mode = "manual"

            try:
                popef_name = f"sharded_model_{current_process().name}_{popef_idx}.popef"
                popef_idx += 1
                Compiler.compile_and_export(sharded_model, outputs, popef_name, options)
                print(
                    f"[Process {current_process().name}] Compiled successfully and save PopEF {popef_name}."
                )
                compile_results_queue.put(
                    (True, subgraph_groups, perf_idx, popef_name, sharded_model)
                )

            except RuntimeError as e:
                print(f"[Process {current_process().name}] Compile failed.")
                tile_id, out_of_memory = re.findall(
                    r"Out of memory on tile (.+?) bytes", str(e)
                )[0].split(": ")
                compile_results_queue.put(
                    (
                        False,
                        subgraph_groups,
                        perf_idx,
                        int(tile_id),
                        int(out_of_memory),
                    )
                )

            except Exception as e:
                raise e

        popef_idx = 0
        while True:
            with block_flags.get_lock():
                block_flags[int(current_process().name)] = 1
            subgraph_groups, perf_idx = queue.get(block=True)
            with block_flags.get_lock():
                block_flags[int(current_process().name)] = 0
            # None is the signal of the end of the queue
            if subgraph_groups is None:
                print(f"Exiting in process {current_process().name}")
                break
            compile(subgraph_groups, popef_idx, perf_idx)
            popef_idx += 1

    def _print_sharding_info(self, subgraph_groups: List[SubgraphGroup]) -> None:
        """Print the sharding information of the subgraph groups.

        :param subgraph_groups: the list of SubgraphGroup.
        """
        for i in range(len(subgraph_groups) - 1):
            print(
                f"Sharding nodes: Device {i} - {[s.start_node for s in subgraph_groups[i].start_subgraphs]}"
            )

    def _optimize(
        self, onnx_model: onnx.ModelProto, subgraph_groups: List[SubgraphGroup]
    ):
        """Try to find the sharding nodes which is able to make graph compile successfully.

        :param onnx_model: the ONNX model.
        :param subgraph_groups: the list of SubgraphGroup.
        """
        oom = 0
        oom_tile_id = 0
        cur_best_sg = None
        travelled_groups = []

        # Record the subgraph groups which are able to compile
        valid_subgraph_groups = []

        def stop_workers(workers, queue):
            # Clean queue and perf_queue
            while not queue.empty():
                queue.get()

            # Add the signal of ending for each process
            for _ in range(len(workers)):
                queue.put((None, None))

            for w in workers:
                w.join()

        def skip_traversed(
            combination: List[List[SubgraphGroup]], travelled_groups: List[Set[int]]
        ) -> List[List[SubgraphGroup]]:
            # Skip the traversed combination
            remained = []
            for sg in combination:
                group_start_ids = []
                for g in sg:
                    group_start_ids.append(set([s.id for s in g.start_subgraphs]))

                if group_start_ids not in travelled_groups:
                    travelled_groups.append(group_start_ids)
                    remained.append(sg)
            return remained

        # The queue is used to store the candidate subgraph groups
        queue = Queue()
        # The queue is used to store the compilation of subgraph groups
        compile_results_queue = Queue()
        block_flags = Array('i', [0] * self._num_processes)

        # The queue is used to store the subgraph groups sorted by the FLOPs cost
        perf_queue = []

        queue.put((subgraph_groups, None))
        skip_traversed([subgraph_groups], travelled_groups)
        # Launch the processes
        print("Processes started...")
        workers = []
        for i in range(self._num_processes):
            p = Process(
                target=self._parallel_compile,
                name=str(i),
                args=(
                    queue,
                    compile_results_queue,
                    block_flags,
                    onnx_model,
                ),
            )
            p.start()
            workers.append(p)

        while True:
            try:
                compile_result = compile_results_queue.get(block=False)
                # compile_result[0]: compile successfully or not
                # Successful:
                # compile_result[1]: subgraph groups
                # compile_result[2]: the index of the perf_queue
                # compile_result[3]: popef
                # compile_result[4]: sharded ONNX model

                # Failed:
                # compile_result[1]: subgraph groups
                # compile_result[2]: the index of the perf_queue
                # compile_result[3]: OOM tile id
                # compile_result[4]: OOM size

                compilable = compile_result[0]
                sg = compile_result[1]
                perf_idx = compile_result[2]

                if compilable:
                    popef = compile_result[3]
                    sharded_onnx_model = compile_result[4]

                    if not self._optimal_perf:
                        stop_workers(workers, queue, queue)

                        latency, tput = self._execute(popef)
                        print(
                            "[Process Main] The following sharding solution is compilable:"
                        )
                        self._print_sharding_info(sg)
                        print(f"Latency {latency} ms, Tput {tput}.")
                        return sharded_onnx_model
                    else:
                        # Only keep the top 5 FLOPs-balanced subgraph groups
                        more_balanced = False
                        if len(valid_subgraph_groups) < 5:
                            j = 0
                            while j < len(valid_subgraph_groups):
                                valid_stdv = np.std(
                                    [
                                        s.subgraph_flops_cost
                                        for s in valid_subgraph_groups[j][
                                            "subgraph_groups"
                                        ]
                                    ]
                                )
                                new_stdv = np.std([s.subgraph_flops_cost for s in sg])
                                if new_stdv < valid_stdv:
                                    valid_subgraph_groups.insert(
                                        j, {"subgraph_groups": sg, "popef": popef}
                                    )
                                    break
                                else:
                                    j += 1
                            if j == len(valid_subgraph_groups):
                                valid_subgraph_groups.append(
                                    {"subgraph_groups": sg, "popef": popef}
                                )
                            more_balanced = True
                        else:
                            j = 0
                            while j < len(valid_subgraph_groups):
                                valid_stdv = np.std(
                                    [
                                        s.subgraph_flops_cost
                                        for s in valid_subgraph_groups[j][
                                            "subgraph_groups"
                                        ]
                                    ]
                                )
                                new_stdv = np.std([s.subgraph_flops_cost for s in sg])
                                if new_stdv < valid_stdv:
                                    valid_subgraph_groups.insert(
                                        j, {"subgraph_groups": sg, "popef": popef}
                                    )
                                    valid_subgraph_groups.pop(-1)
                                    more_balanced = True
                                    break
                                j += 1

                        if not more_balanced:
                            continue

                        max_flops_idx = 0
                        for idx in range(len(sg)):
                            if (
                                sg[idx].subgraph_flops_cost
                                > sg[max_flops_idx].subgraph_flops_cost
                            ):
                                max_flops_idx = idx
                        combination = self._remove_to_fit(max_flops_idx, sg, True)
                        combination = skip_traversed(combination, travelled_groups)
                        if not combination:
                            continue

                        flops_stdv = []
                        for i in range(len(combination)):
                            flops_stdv.append(
                                np.std([s.subgraph_flops_cost for s in combination[i]])
                            )
                        sorted_idx = sorted(
                            range(len(flops_stdv)), key=lambda k: flops_stdv[k]
                        )
                        sorted_combination = [combination[idx] for idx in sorted_idx]
                        # Skip the less FLOPs-balanced subgraph groups
                        for v in valid_subgraph_groups:
                            valid_sg = v["subgraph_groups"]
                            i = 0
                            while i < len(sorted_combination):
                                new_sg = sorted_combination[i]
                                valid_stdv = np.std(
                                    [s.subgraph_flops_cost for s in valid_sg]
                                )
                                new_stdv = np.std(
                                    [s.subgraph_flops_cost for s in new_sg]
                                )
                                if new_stdv > valid_stdv:
                                    sorted_combination.pop(i)
                                else:
                                    i += 1

                        if perf_idx is None:
                            if not sorted_combination:
                                continue
                            # Add new combination to the perf_queue
                            perf_queue.append(sorted_combination)
                            perf_idx = len(perf_queue) - 1
                            queue.put((perf_queue[perf_idx][0], perf_idx))
                            perf_queue[perf_idx].pop(0)
                        else:
                            if not sorted_combination:
                                perf_queue[perf_idx] = sorted_combination
                                continue

                            if not perf_queue[perf_idx]:
                                continue

                            # Update the exist combination
                            perf_queue[perf_idx] = sorted_combination
                            queue.put((perf_queue[perf_idx][0], perf_idx))
                            perf_queue[perf_idx].pop(0)

                elif (
                    self._optimal_perf
                    and perf_queue
                    and perf_idx is not None
                    and perf_queue[perf_idx]
                ):
                    # If the most FLOPs-balanced subgraph groups is not compilable,
                    # going to try next less FLOPs-balanced subgraph groups.
                    queue.put((perf_queue[perf_idx][0], perf_idx))
                    perf_queue[perf_idx].pop(0)

                else:
                    tile_id = compile_result[3]
                    memory = compile_result[4]
                    device_id = tile_id // 1472
                    # Each IPU tile has 638976 bytes of memory.
                    # If the memory usage of the current SubgraphGroup list is close to 638976 bytes,
                    # we can try to traverse updated SubgraphGroup list from the current list.
                    # In case of skipping the best solution.
                    if oom < 641000 or oom >= memory:
                        combination = self._remove_to_fit(device_id, sg)
                        for new_sg in skip_traversed(combination, travelled_groups):
                            queue.put((new_sg, None))

                    if oom == 0 or oom >= memory:
                        oom = memory
                        oom_tile_id = tile_id
                        cur_best_sg = copy.deepcopy(sg)
            except Empty:
                real_stop = True
                # Check if all processes are blocked
                for _ in range(5):
                    with block_flags.get_lock():
                        for i in range(len(block_flags)):
                            if block_flags[i] == 0:
                                real_stop = False
                                break
                    if real_stop:
                        time.sleep(10)
                    else:
                        continue

                if real_stop and queue.empty() and compile_results_queue.empty():
                    stop_workers(workers, queue)
                    break

            except KeyboardInterrupt:
                # Ctrl + C terminate all processes
                print("Terminating worker...")
                for w in workers:
                    w.terminate()
                    w.join()

        if valid_subgraph_groups:
            optimal_idx = 0
            optimal_latency = 0
            optimal_tput = 0
            for i in range(len(valid_subgraph_groups)):
                latency, tput = self._execute(valid_subgraph_groups[i]["popef"])
                if i == 0:
                    optimal_latency = latency
                    optimal_tput = tput
                else:
                    if latency < optimal_latency:
                        optimal_latency = latency
                        optimal_tput = tput
                        optimal_idx = i
            print("[Success] The optimal solution: ")
            self._print_sharding_info(
                valid_subgraph_groups[optimal_idx]["subgraph_groups"]
            )
            print(
                f"[Success] The optimal latency {optimal_latency} ms, tput {optimal_tput}"
            )
            sharding_info = {}
            for i in range(len(valid_subgraph_groups[optimal_idx]["subgraph_groups"])):
                for start_subgraph in valid_subgraph_groups[optimal_idx][
                    "subgraph_groups"
                ][i].start_subgraphs:
                    sharding_info[start_subgraph.start_node] = i

            return Sharder(sharding_info).run(onnx_model)

        # Try to lower the available_memory_proportion
        print("Try to lower the available_memory_proportion...")
        for amp in [0.3, 0.1]:
            sharding_info = {}
            for i in range(len(cur_best_sg)):
                for start_subgraph in cur_best_sg[i].start_subgraphs:
                    sharding_info[start_subgraph.start_node] = i
            sharded_model = Sharder(sharding_info).run(copy.deepcopy(onnx_model))
            outputs = [o.name for o in sharded_model.graph.output]

            options = CompilerOptions()
            options.ipu_version = runtime.DeviceManager().ipu_hardware_version()
            options.num_ipus = self._num_ipus
            options.virtual_graph_mode = "manual"
            options.available_memory_proportion = amp

            try:
                popef = f"sharded_model_with_amp_{amp}.popef"
                Compiler.compile_and_export(sharded_model, outputs, popef, options)
                print(
                    f"[Success] Compile successfully available_memory_proportion {amp} and solution:"
                )
                self._print_sharding_info(cur_best_sg)
                latency, tput = self._execute(popef)
                print(f"[Success] The latency {latency} ms, tput {tput}")
                return sharded_model
            except RuntimeError:
                print(
                    f"[Failed] Failed to compile with available_memory_proportion {amp}"
                )

        print(f"[Failed] The model cannot be sharded into {args.num_ipus} IPUs")
        print(f"[Failed] The current best sharding with {args.num_ipus} IPUs is:")
        self._print_sharding_info(cur_best_sg)
        print(f"[Failed] Out of memory on tile {oom_tile_id}: {oom} bytes")
        return None

    def run(self, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Run auto-sharding.

        :param onnx_model: The original ModelProto.
        """
        print("-------- Auto-sharding start -------")
        # Check if each node has unique name
        node_list = []
        for n in onnx_model.graph.node:
            if n.name == "":
                raise ValueError(
                    f"Each node need to have an unique name. The node has no name: \n{n}"
                )
            if n.name in node_list:
                raise ValueError(
                    f"Each node need to have an unique name. The node has duplicated name: \n{n}"
                )
            node_list.append(n.name)

        self._init_info(onnx_model.graph)

        avg_bytes_cost = self._total_bytes_cost / len(self._candidates)
        avg_flops_cost = self._total_flops_cost / len(self._candidates)
        # All subgraphs from candidate nodes
        for start_node in self._candidates:
            subgraph_nodes = self._traverse_subgraph(start_node)
            if subgraph_nodes:
                subgraph_node_protos = [
                    self._name_to_node_proto[node] for node in subgraph_nodes
                ]
                bytes_cost = self._bytes_cost_by_nodes(subgraph_node_protos)
                flops_cost = self._flops_cost_by_nodes(subgraph_node_protos)
                # Skip the subgraph if the bytes cost or flops cost is too small
                if bytes_cost < avg_bytes_cost or flops_cost < avg_flops_cost:
                    for n in subgraph_nodes:
                        self._travelled.remove(n)
                    continue
                subgraph_id = len(self._all_subgraphs)
                subgraph = Subgraph(
                    start_node,
                    list(subgraph_nodes),
                    subgraph_id,
                    bytes_cost,
                    flops_cost,
                )
                # Find parents and kids of the subgraph
                for s in self._all_subgraphs:
                    for subgraph_node in subgraph_nodes:
                        if s.start_node in self._node_to_input_nodes[subgraph_node]:
                            subgraph.parents.add(s.id)
                            s.kids.add(subgraph.id)
                            subgraph.included_subgraphs.add(s.id)
                            subgraph.included_subgraphs.union(s.included_subgraphs)
                self._all_subgraphs.append(subgraph)

        # Regard remained nodes as the last subgraph
        remained_nodes = []
        remained_node_protos = []
        for n in onnx_model.graph.node:
            if n.name not in self._travelled:
                remained_nodes.append(n.name)
                remained_node_protos.append(n)
        if remained_nodes:
            bytes_cost = self._bytes_cost_by_nodes(remained_node_protos)
            flops_cost = self._flops_cost_by_nodes(remained_node_protos)
            subgraph_id = len(self._all_subgraphs)
            remained_subgraph = Subgraph(
                "Remained",
                remained_nodes,
                subgraph_id,
                bytes_cost,
                flops_cost,
            )
            for s in self._all_subgraphs:
                for subgraph_node in remained_nodes:
                    if s.start_node in self._node_to_input_nodes[subgraph_node]:
                        remained_subgraph.parents.add(s.id)
            self._all_subgraphs.append(remained_subgraph)

        for i in range(len(self._all_subgraphs)):
            for j in range(i + 1, len(self._all_subgraphs)):
                if self._all_subgraphs[i].is_parallel(self._all_subgraphs[j]):
                    self._all_subgraphs[i].parallel_subgraphs.add(j)
                    self._all_subgraphs[j].parallel_subgraphs.add(i)

        print(f"The number of subgraphs: {len(self._all_subgraphs)}")
        subgraph_groups = self._raw_sharding()
        return self._optimize(onnx_model, subgraph_groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='auto sharding',
        description='Try to auto shard a ONNX model into devices and pipelining stages.',
        formatter_class=PopHelpFormatter,
    )
    parser.add_argument(
        '--input_model',
        type=str,
        help='Set the path of the original ONNX model.',
    )
    parser.add_argument(
        '--num_ipus',
        type=int,
        help='Set the number of IPUs.',
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default=None,
        help='Set the name of the converted ONNX model.',
    )
    parser.add_argument(
        '--optimal_perf',
        action='store_true',
        help='Travese to find the best performance sharding solution.',
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=2,
        help='Set the number of processes for compilation.',
    )
    args = parser.parse_args()

    # input_model
    if not os.path.isfile(args.input_model):
        raise FileNotFoundError(f"{args.input_model} not found")
    input_model = args.input_model

    # output_model
    if args.output_model:
        output_model = args.output_model
    else:
        output_model = input_model + '.auto_sharded.onnx'

    ori_onnx_model = onnx.load(input_model)

    # Remove POPLAR_ENGINE_OPTIONS
    if 'POPLAR_ENGINE_OPTIONS' in os.environ:
        print(
            "WARNING: POPLAR_ENGINE_OPTIONS will be unset during compilation process."
        )
        del os.environ['POPLAR_ENGINE_OPTIONS']

    # auto sharding
    if args.num_ipus <= 1:
        raise ValueError("The number of IPUs must be greater than 1.")
    sharded_onnx_model = AutoSharding(
        args.num_ipus, args.optimal_perf, args.num_processes
    ).run(ori_onnx_model)

    if sharded_onnx_model:
        onnx.save(sharded_onnx_model, output_model)
        print(f"[Success] Save the sharded model to {output_model}")
print("-------- Auto-sharding end -------")
