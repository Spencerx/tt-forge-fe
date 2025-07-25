// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ops/op.hpp"
#pragma once

namespace tt::graphlib
{
class Graph;
class OpNode;
class Node;
class Shape;
struct OpType;
struct Edge;
}  // namespace tt::graphlib

namespace tt::passes
{

int volume_above(std::vector<std::uint32_t> shape, int dim);
int volume_below(std::vector<std::uint32_t> shape, int dim);
std::tuple<bool, int> can_commute_reshape_through_dim(
    graphlib::Shape input_shape, graphlib::Shape output_shape, int dim, bool commute_up = false);
std::tuple<bool, int> can_commute_through_dim(
    graphlib::OpNode *initial_op, graphlib::Graph *graph, int dim, bool commute_up = false);

bool match_unsqueeze(graphlib::OpType const &a, graphlib::OpType const &b);

bool match_squeeze(graphlib::OpType const &a, graphlib::OpType const &b);

bool match_reshape(graphlib::OpType const &a, graphlib::OpType const &);

bool match_transpose(graphlib::OpType const &a, graphlib::OpType const &b);

using MatchFn = bool(graphlib::OpType const &, graphlib::OpType const &);
static std::unordered_map<ops::OpType, MatchFn *> match_fns = {
    {ops::OpType::Reshape, match_reshape},
    {ops::OpType::Transpose, match_transpose},
    {ops::OpType::Unsqueeze, match_unsqueeze},
    {ops::OpType::Squeeze, match_squeeze},
};

size_t total_broadcast_volume(graphlib::Graph *graph, graphlib::Edge edge);
std::pair<bool, int> are_inverse_with_broadcast(
    graphlib::Shape shape_a, graphlib::Shape shape_b, size_t broadcast_volume);
bool are_compatible_ops(
    graphlib::Graph *graph,
    graphlib::OpNode *a,
    graphlib::OpNode *b,
    graphlib::Shape *updated_shape = nullptr,
    bool check_inverse = true);
graphlib::Shape shape_of_only_operand(graphlib::Graph *graph, graphlib::OpNode *op);

bool commute_through_concat(
    graphlib::Graph *graph,
    graphlib::OpNode *op,
    graphlib::OpNode *initial_op,
    graphlib::Node *producer,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape,
    bool check_only,
    bool *retain_operand_dim,
    std::pair<int, int> *operand_dims,
    graphlib::OpType *golden_transform,
    bool commute_up = false);

bool can_commute_through_concat(
    graphlib::Graph *graph,
    graphlib::OpNode *op,
    graphlib::OpNode *initial_op,
    graphlib::Node *producer,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape,
    bool commute_up);

bool commute_through_select(
    graphlib::Graph *graph,
    graphlib::OpNode *op,
    graphlib::OpNode *initial_op,
    graphlib::Node *producer,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape,
    bool check_only,
    bool *retain_operand_dim,
    std::pair<int, int> *operand_dims,
    graphlib::OpType *golden_transform,
    bool commute_up = false);

bool can_commute_through_select(
    graphlib::Graph *graph,
    graphlib::OpNode *op,
    graphlib::OpNode *initial_op,
    graphlib::Node *producer,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape,
    bool commute_up);

bool commute_through_reduce(
    graphlib::Graph *graph,
    graphlib::OpNode *op,
    graphlib::OpNode *initial_op,
    graphlib::OpNode *producer,
    graphlib::Node *next,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape,
    bool check_only,
    bool *retain_operand_dim,
    std::pair<int, int> *operand_dims,
    graphlib::OpType *golden_transform,
    bool commute_up = false);

bool can_commute_through_reduce(
    graphlib::Graph *graph,
    graphlib::OpNode *op,
    graphlib::OpNode *initial_op,
    graphlib::OpNode *producer,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape,
    bool commute_up);

bool commute_through_eltwise(
    graphlib::OpNode *op, graphlib::Shape *commute_shape = nullptr, graphlib::OpType *golden_transform = nullptr);

bool commute_through_quantization(
    graphlib::OpNode *op, graphlib::Shape *commute_shape = nullptr, graphlib::OpType *golden_transform = nullptr);

bool is_quantization_ops(graphlib::OpNode *op);

bool can_commute_past_op(
    graphlib::OpNode *op,
    graphlib::OpNode *initial_op,
    graphlib::Graph *graph,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape = nullptr,
    bool commute_up = false,
    graphlib::Node *producer = nullptr);

void update_reshape_attr(graphlib::OpNode *reshape, graphlib::Shape new_shape);
void update_select_attr(
    graphlib::OpNode *select_op,
    int select_dim,
    std::optional<int> begin = std::nullopt,
    std::optional<int> length = std::nullopt,
    std::optional<int> stride = std::nullopt);
void update_concat_attr(graphlib::OpNode *op, int new_dim);
void update_reduce_attr(graphlib::OpNode *reduce, int reduce_dim, bool keep_dim);
void update_matmul_attr(graphlib::OpNode *matmul, int requant_zp);
void update_conv_attr(graphlib::OpNode *conv, const std::vector<int> &pad_attrs);
void update_vstack_attr(graphlib::OpNode *vstack, int new_value);

std::pair<bool, std::pair<std::vector<int>, std::vector<int>>> handle_shape_change_through_bcast(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::Node *producer,
    graphlib::OpNode *consumer,
    graphlib::Shape *commute_shape,
    graphlib::Shape *clone_shape);

bool try_commute_bcast_through_clone(graphlib::Graph *graph, graphlib::OpNode *node);

bool all_producer_forks_have_equivalent(
    graphlib::Graph *graph,
    graphlib::OpNode *initial_op,
    graphlib::Shape commute_shape,
    graphlib::OpNode *from = nullptr);

}  // namespace tt::passes
