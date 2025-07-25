// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <string>
#include <vector>

#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"

namespace at
{
class Tensor;
}

namespace py = pybind11;

namespace tt
{

namespace autograd
{

struct autograd_config
{
    bool recompute = false;  // Add recompute
    py::object optimizer = py::none();
};

using grad_map = std::unordered_map<tt::graphlib::EdgeUniqueId, bool, EdgeUniqueIdHash>;

using Node = graphlib::Node;
using Graph = graphlib::Graph;
using NodeContext = graphlib::NodeContext;

class autograd_engine
{
   private:
    tt::graphlib::Graph *graph;
    autograd_config config;

    // fwd->output gradient producer map
    std::unordered_map<Node *, std::vector<Node *>> fwd_to_out_gradient_map;

   public:
    autograd_engine(Graph *graph, autograd_config config);
    ~autograd_engine() = default;
    autograd_engine(const autograd_engine &other) = delete;

    // Run and return the modified graph
    Graph *run();

    // Create an op for the given fwd op's operand.
    NodeContext create_op(
        struct autograd_context &self, const graphlib::OpType &type, const std::vector<NodeContext> &operands);

    // Create a backward op for the given fwd op's operand.
    NodeContext create_backward_op(
        const graphlib::OpType &type,
        const std::vector<NodeContext> &operands,
        Node *current_fwd_op,
        int operand_index,
        int created_op_index,
        std::string name_prefix = "",
        bool copy_golden_transforms = true);

    NodeContext create_optimizer_op(
        const graphlib::OpType &type,
        const std::vector<NodeContext> &operands,
        Node *current_fwd_op,
        int operand_index,
        int created_op_index,
        std::string name_prefix = "");

    template <typename T>
    NodeContext create_constant(struct tt::autograd::autograd_context &self, T value);

    // Create an integer constant used in backward calculations (typically a negative one)
    template <typename T>
    NodeContext create_constant(
        Node *current_fwd_op, int operand_index, T value, int created_op_index, graphlib::NodeEpochType epoch_type)
    {
        auto node = graph->add_node(
            graphlib::create_node<graphlib::ConstantInputNode>(
                "input_constant_" + current_fwd_op->name() + "_" + std::to_string(created_op_index), value),
            graph->get_subgraph_id_for_node(current_fwd_op->id()));

        node->set_shape(graphlib::Shape::create({1}));
        node->set_output_df(current_fwd_op->output_df());

        if (epoch_type == graphlib::NodeEpochType::Backward)
        {
            node->set_backward();
            add_fwd_to_bwd_map(current_fwd_op, node, operand_index);
        }
        else if (epoch_type == graphlib::NodeEpochType::Optimizer)
        {
            node->set_optimizer();
            add_fwd_to_optimizer_edge(current_fwd_op, node, operand_index);
        }

        return NodeContext(node);
    }

    NodeContext create_constant_tensor(
        Node *current_fwd_op,
        int operand_index,
        const at::Tensor &tensor,
        int created_op_index,
        graphlib::NodeEpochType epoch_type);

    NodeContext create_constant_tensor(struct autograd_context &self, const at::Tensor &tensor);

    NodeContext create_input(
        Node *current_fwd_op,
        int operand_index,
        int created_op_index,
        graphlib::NodeEpochType epoch_type,
        std::string &suffix_identifier,
        const std::vector<std::uint32_t> &tensor_shape,
        bool copy_consteval_operations,
        bool disable_consteval = false);

    bool contains_bwd_nodes() const;
    const std::map<int, std::vector<Node *>> &get_bwd_nodes(Node *fwd) const;

    // Get pointer to graph being worked on
    Graph *get_graph() const { return graph; }

   private:
    // Propagate requires_grad from inputs to all edges of the graph, creating an edge->bool map
    grad_map propagate_requires_grad();

    // Create backward instructions, and hook them up accordingly
    void create_backward_graph(const grad_map &requires_grad_map);

    // Register fwd->bwd and bwd->fwd relationship
    void add_fwd_to_bwd_map(Node *fwd, Node *bwd, int operand_index, bool gradient = false);

    void add_fwd_to_optimizer_edge(Node *fwd, Node *opt, int operand_index);

    // Register fwd->out_gradient
    void add_fwd_to_out_gradient_map(Node *fwd, Node *out_gradient);

    // Combine incoming gradients by adding them, and return the new combined node
    Node *combine_incoming_gradients(Node *node);

    // Create optinstructions, and hook them up accordingly
    void create_optimizer_graph();

    // Inserts ops in the backward graph that recompute the whole forward graph.
    // This can be used to avoid the need to store all of the intermediate tensors to be able to run the backward pass.
    void insert_recompute_ops();
};

// Structure passed to python while generating backward ops. This allows us to register
// backward ops in both the graph and autograd engine maps
struct __attribute__((visibility("hidden"))) autograd_context
{
    autograd_engine *autograd;
    Node *current_fwd_op;
    int operand;
    graphlib::NodeEpochType epoch_type = graphlib::NodeEpochType::Backward;
    int created_op_index = 0;  // Incremented to ensure unique names when multiple ops are created
};

template <typename T>
NodeContext autograd_engine::create_constant(struct tt::autograd::autograd_context &self, T value)
{
    return self.autograd->create_constant<T>(
        self.current_fwd_op, self.operand, value, self.created_op_index++, self.epoch_type);
}

}  // namespace autograd
}  // namespace tt
