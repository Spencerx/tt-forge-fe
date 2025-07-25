// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/squeeze_to_reshape.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/commute_utils.hpp"

namespace tt::passes
{
bool squeeze_to_reshape(graphlib::Graph *graph)
{
    bool changed_anything = false;
    for (auto *node : graphlib::topological_sort(*graph))
    {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
        if (not op)
            continue;

        if (op->new_op_type() != ops::OpType::Squeeze and op->new_op_type() != ops::OpType::Unsqueeze)
            continue;

        std::vector<uint32_t> shape_vec = op->shape().as_vector();
        std::vector<graphlib::OpType::Attr> shape;

        for (uint32_t d : shape_vec)
        {
            shape.push_back((int)d);
        }

        op->change_op_type("reshape");
        graphlib::Shape new_shape = graphlib::Shape::create(shape_vec);
        update_reshape_attr(op, new_shape);
        changed_anything = true;
    }
    return changed_anything;
}
}  // namespace tt::passes
