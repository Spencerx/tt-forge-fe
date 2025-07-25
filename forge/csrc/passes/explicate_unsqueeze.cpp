// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/explicate_unsqueeze.hpp"

#include <pybind11/pybind11.h>

#include <functional>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/passes_utils.hpp"
#include "passes/print_graph.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

void explicate_unsqueeze(graphlib::Graph *graph)
{
    // Insert explicit unsqueeze op for eltwise binary ops
    for (auto *node : graphlib::topological_sort(*graph))
    {
        auto op = dynamic_cast<graphlib::OpNode *>(node);

        if (not op)
        {
            continue;
        }
        if (op->is_eltwise_binary())
        {
            auto operand_a = graph->operands(node)[0];
            auto operand_b = graph->operands(node)[1];
            if (operand_a->shape().size() == operand_b->shape().size())
            {
                continue;
            }

            bool operand_a_is_input = dynamic_cast<graphlib::InputNode const *>(operand_a) != NULL;
            bool operand_b_is_input = dynamic_cast<graphlib::InputNode const *>(operand_b) != NULL;
            if ((operand_a->shape().size() < operand_b->shape().size() and operand_a_is_input) or
                (operand_a->shape().size() > operand_b->shape().size() and operand_b_is_input))
            {
                continue;
            }

            auto insert = [graph](graphlib::Node *to_be_unsqueeze, graphlib::Node *reference, graphlib::Node *eltwise)
            {
                auto current_node = to_be_unsqueeze;
                while (current_node->shape().size() < reference->shape().size())
                {
                    auto rank = current_node->shape().size();
                    std::string name = to_be_unsqueeze->name() + "_" + eltwise->name() + "_unsqueeze_" +
                                       std::to_string(rank) + "_operand_0";
                    auto attr = std::vector<graphlib::OpType::Attr>{0, 0};
                    auto named_attr = graphlib::OpType::Attrs{{"dim", 0}};
                    auto op_type = graphlib::OpType("unsqueeze", attr, named_attr);
                    auto change_rank = graph->add_node(
                        std::make_unique<graphlib::PyOpNode>(name, op_type),
                        graph->get_subgraph_id_for_node(current_node->id()));

                    // graphlib::OpNode *change_rank = dynamic_cast<graphlib::OpNode
                    // *>(graph->add_node(current_node->clone(name))); TT_ASSERT(change_rank);
                    change_rank->set_shape(current_node->shape().as_rank(rank + 1));
                    change_rank->set_output_df(to_be_unsqueeze->output_df());

                    auto current_edge = graph->get_edges(current_node, eltwise)[0];
                    auto current_tms = graph->get_edge_attributes(current_edge)->get_tms();
                    auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, current_edge, change_rank);
                    graph->get_edge_attributes(incoming_edge)->set_tms({});
                    graph->get_edge_attributes(outgoing_edge)->set_tms(current_tms);
                    current_node = change_rank;
                }
            };

            // Add explicit unsqueeze op
            if (operand_a->shape().size() < operand_b->shape().size())
            {
                insert(operand_a, operand_b, node);
            }
            else
            {
                insert(operand_b, operand_a, node);
            }
            recalculate_shapes(graph);
        }
    }
}

void hoist_unsqueeze_squeeze_to_reshape(graphlib::Graph *graph)
{
    std::unordered_set<graphlib::Node *> nodes_to_remove;
    for (auto *node : graphlib::topological_sort(*graph))
    {
        auto op = dynamic_cast<graphlib::OpNode *>(node);

        if (not op)
        {
            continue;
        }
        if (op->new_op_type() != ops::OpType::Reshape)
        {
            continue;
        }
        // Find reshape -> unsqueeze pattern and replace with reshape
        auto users = graph->users(node);
        auto operands = graph->operands(node);
        if (users.size() != 1 or operands.size() != 1)
        {
            continue;
        }

        auto user_op = dynamic_cast<graphlib::OpNode *>(users[0]);
        auto operand_op = dynamic_cast<graphlib::OpNode *>(operands[0]);
        bool user_is_squeeze_unsqueeze =
            (user_op and
             (user_op->new_op_type() == ops::OpType::Unsqueeze or user_op->new_op_type() == ops::OpType::Squeeze));
        bool operand_is_squeeze_unsqueeze =
            (operand_op and (operand_op->new_op_type() == ops::OpType::Unsqueeze or
                             operand_op->new_op_type() == ops::OpType::Squeeze));
        if (not user_is_squeeze_unsqueeze and not operand_is_squeeze_unsqueeze)
        {
            continue;
        }

        if (user_is_squeeze_unsqueeze)
        {
            auto target_shape = user_op->shape().as_vector();
            std::vector<graphlib::OpType::Attr> new_reshape_attr;
            for (auto dim : target_shape)
            {
                new_reshape_attr.push_back((int)dim);
            }
            std::vector<int> shape_vector(target_shape.begin(), target_shape.end());
            graphlib::OpType::Attrs named_attrs;
            named_attrs["shape"] = shape_vector;
            op->change_op_type(graphlib::OpType("reshape", new_reshape_attr, named_attrs));
            op->set_shape(user_op->shape());
            nodes_to_remove.insert(users[0]);
        }

        if (operand_is_squeeze_unsqueeze)
        {
            nodes_to_remove.insert(operands[0]);
        }
    }

    for (auto node : nodes_to_remove)
    {
        auto maintain_tms = [graph](graphlib::Edge new_edge, graphlib::Edge original_edge)
        {
            auto original_tms = graph->get_edge_attributes(original_edge)->get_tms();
            auto new_tms = graph->get_edge_attributes(new_edge)->get_tms();
            new_tms.insert(new_tms.end(), original_tms.begin(), original_tms.end());
        };
        bypass_node(graph, node, true, maintain_tms);
    }
    recalculate_shapes(graph);
}

}  // namespace tt::passes
