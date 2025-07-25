// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "passes/fuse_conv2d_bias.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "ops/op.hpp"
#include "python_bindings_common.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

static bool has_fusable_upstream_conv2d(graphlib::Graph *graph, graphlib::PyOpNode *op)
{
    if (op == nullptr)
        return false;

    if (op->new_op_type() != ops::OpType::Conv2d)
        return false;
    // If conv2d has more outputs than just to bias, we can't merge
    if (graph->user_data_edges(op).size() > 1)
        return false;

    // If conv2d already has bias merged, we can't merge another one
    if (graph->operand_data_edges(op).size() > 2)
        return false;

    return true;
}

void fuse_conv2d_bias(graphlib::Graph *graph)
{
    for (tt::graphlib::Node *node : graphlib::topological_sort(*graph))
    {
        // Look for bias
        if ((node->node_type() != graphlib::kPyOp) ||
            (node->as<graphlib::PyOpNode>()->new_op_type() != ops::OpType::Add))
            continue;

        graphlib::PyOpNode *op = node->as<graphlib::PyOpNode>();

        auto operands = graph->data_operands(op);
        TT_ASSERT(operands.size() == 2);

        if (not has_fusable_upstream_conv2d(graph, dynamic_cast<graphlib::PyOpNode *>(operands[0])))
            continue;

        auto tms = graph->get_edge_attributes(graph->operand_data_edges(op)[1])->get_tms();
        bool broadcast = false;
        for (auto tm : tms)
            if (tm.type() == ops::OpType::Broadcast)
            {
                broadcast = true;
                break;
            }

        if (!broadcast)
            continue;

        log_trace(LogGraphCompiler, "Merging {} and {}", operands[0]->name(), op->name());

        // Create a new bias edge to conv2d
        tt::graphlib::Edge bias_input_edge = graph->operand_data_edges(op)[1];

        // The broadcast dimensions are cleared in the bias edge before copying to the conv2d edge,
        // as the conv2d bias needs to be converted to the required layout.
        graph->get_edge_attributes(bias_input_edge)->clear_broadcast_dims();

        tt::graphlib::Edge new_bias_input_edge = tt::graphlib::Edge(
            bias_input_edge.producer_node_id,
            bias_input_edge.producer_output_port_id,
            operands[0]->id(),
            2,
            graphlib::EdgeType::kData);
        graph->add_edge(new_bias_input_edge);
        graph->copy_edge_attributes(bias_input_edge, new_bias_input_edge);

        // Get user edges for Add, and copy over to conv2d
        auto user_edges = graph->user_edges(op);
        for (tt::graphlib::Edge edge : user_edges)
        {
            graphlib::PortId producer_output_port_id = 0;
            if (edge.edge_type == graphlib::EdgeType::kAutogradFwdToBwd)
            {
                producer_output_port_id = 2;
            }
            tt::graphlib::Edge new_edge = tt::graphlib::Edge(
                operands[0]->id(),
                producer_output_port_id,
                edge.consumer_node_id,
                edge.consumer_input_port_id,
                edge.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(edge, new_edge);
        }

        // The output of fused conv should match the output of the original add
        operands[0]->as<graphlib::OpNode>()->set_golden_transforms(op->get_golden_transforms());
        operands[0]->as<graphlib::OpNode>()->set_golden_id(op->id());

        // Remove add
        graph->remove_node(op);
    }
}

}  // namespace tt::passes
