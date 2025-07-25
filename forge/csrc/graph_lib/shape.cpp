// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/shape.hpp"

#include <sstream>
#include <tuple>

#include "graph_lib/utils.hpp"
#include "utils/assert.hpp"
namespace tt
{
namespace graphlib
{

int Shape::get_tile_height() const { return tt::graphlib::get_row_size_from_tile_size(tile_dim_); }

int Shape::get_tile_width() const { return tt::graphlib::get_col_size_from_tile_size(tile_dim_); }

Shape::Shape(bool valid, Shape::Type type, std::vector<std::uint32_t> dims) :
    valid_(valid), type_(type), dims_(std::move(dims))
{
}

Shape Shape::create_forge(std::vector<std::uint32_t> values, int tile_height, int tile_width)
{
    TT_ASSERT(
        values.size() >= FORGE_DIM_COUNT and values.size() <= FORGE_MAX_DIM_COUNT, "Shape must be 4/5-dimensional");
    TT_ASSERT(
        values[values.size() - 1] % tile_width == 0,
        "Column dimension must be divisible by tile size",
        Shape::create(values));
    TT_ASSERT(
        values[values.size() - 2] % tile_height == 0,
        "Row dimension must be divisible by tile size",
        Shape::create(values));

    TileDim tile_dim = tt::graphlib::get_tile_dim_from_height_width(tile_height, tile_width);
    Shape s;
    s.dims_ = std::move(values);
    s.valid_ = true;
    s.type_ = FORGE;
    s.tile_dim_ = tile_dim;
    return s;
}

Shape Shape::create_forge(std::uint32_t w, std::uint32_t z, std::uint32_t r, std::uint32_t c)
{
    TT_ASSERT(r % FORGE_TILE_DIM == 0, "Row dimension must be divisible by tile size");
    TT_ASSERT(c % FORGE_TILE_DIM == 0, "Column dimension must be divisible by tile size");
    Shape s;
    s.dims_ = {w, z, r, c};
    s.valid_ = true;
    s.type_ = FORGE;
    return s;
}

Shape Shape::create_with_type_from_other(const Shape &other, std::vector<std::uint32_t> dims)
{
    Shape s;

    s.dims_ = std::move(dims);
    s.valid_ = true;
    s.type_ = FREE;

    if (other.type_ == FORGE)
    {
        s.type_ = FORGE;
        TT_ASSERT(
            s.dims_.size() >= FORGE_DIM_COUNT and s.dims_.size() <= FORGE_MAX_DIM_COUNT,
            "Shape must be 4/5-dimensional");
        TT_ASSERT(s.dims_[-1] % FORGE_TILE_DIM == 0, "Row dimension must be divisible by tile size");
        TT_ASSERT(s.dims_[-2] % FORGE_TILE_DIM == 0, "Column dimension must be divisible by tile size");
    }

    return s;
}

Shape Shape::to_forge(const Shape &other)
{
    if (other.type_ == FORGE)
        return other;

    std::vector<std::uint32_t> dims = other.dims_;

    // Make it at most 5D
    while (dims.size() > FORGE_MAX_DIM_COUNT)
    {
        TT_ASSERT(dims[0] == 1, "Too many dimensions to convert to forge shape");
        dims.erase(dims.begin());
    }
    while (dims.size() < FORGE_DIM_COUNT)
    {
        dims.insert(dims.begin(), 1);
    }

    int forge_tile_r = tt::graphlib::get_row_size_from_tile_size(other.tile_dim_);
    int forge_tile_c = tt::graphlib::get_col_size_from_tile_size(other.tile_dim_);

    // Snap to tile sizes
    size_t dim_r = dims.size() - 2;
    for (std::size_t dim = dim_r; dim <= (dim_r + 1); dim++)
    {
        int current_forge_dim = dim == dim_r ? forge_tile_r : forge_tile_c;
        if (dims[dim] % current_forge_dim != 0)
        {
            dims[dim] += current_forge_dim - (dims[dim] % current_forge_dim);
        }
    }

    Shape s;
    s.dims_ = dims;
    s.valid_ = true;
    s.type_ = FORGE;
    s.tile_dim_ = other.tile_dim_;
    return s;
}

std::uint32_t &Shape::operator[](int i)
{
    if (i < 0)
        i += dims_.size();
    TT_ASSERT(
        (i >= 0) && (i < (int)dims_.size()), "Trying to access element outside of dimensions: " + std::to_string(i));
    TT_ASSERT(valid_, "Shape is not valid_.");
    return dims_[i];
}
std::uint32_t const &Shape::operator[](int i) const
{
    if (i < 0)
        i += dims_.size();
    TT_ASSERT(
        (i >= 0) && (i < (int)dims_.size()), "Trying to access element outside of dimensions: " + std::to_string(i));
    TT_ASSERT(valid_, "Shape is not valid_.");
    return dims_[i];
}

bool Shape::operator==(const Shape &other) const
{
    auto const &a = this->dims_;
    auto const &b = other.dims_;

    int size = (int)std::max(a.size(), b.size());
    for (int i = -1; i >= -size; --i)
    {
        int ai = (int)a.size() + i;
        int bi = (int)b.size() + i;
        if (ai < 0 and b[bi] != 1)
            return false;
        if (bi < 0 and a[ai] != 1)
            return false;
        if (ai < 0 or bi < 0)
            continue;
        if (a[ai] != b[bi])
            return false;
    }

    return true;
}

bool Shape::operator!=(const Shape &other) const { return not(*this == other); }

std::string Shape::as_string() const
{
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &operator<<(std::ostream &out, const Shape &s)
{
    if (!s.is_valid())
        return out << "{INVALID_SHAPE}";

    const std::vector<std::uint32_t> dims_ = s.as_vector();
    TT_ASSERT(dims_.size() > 0);
    out << "{";
    out << s[0];
    for (std::size_t i = 1; i < dims_.size(); ++i)
    {
        out << ", " << dims_[i];
    }
    out << "}";
    return out;
}

std::uint32_t Shape::volume() const
{
    std::uint32_t v = 1;
    for (auto i : dims_) v *= i;
    return v;
}

std::tuple<int, int, int, int> Shape::as_tuple() const
{
    auto can = canonical();
    return std::make_tuple(can[0], can[1], can[2], can[3]);
}

Shape Shape::canonical() const
{
    auto v = as_vector();
    while (v.size() < 4) v.insert(v.begin(), 1);
    return Shape::create(v);
}

Shape Shape::as_rank(std::uint32_t rank) const
{
    TT_ASSERT(rank > 0);
    auto v = as_vector();
    while ((std::uint32_t)v.size() < rank)
    {
        v.insert(v.begin(), 1);
    }
    while ((std::uint32_t)v.size() > rank)
    {
        TT_ASSERT(v.front() == 1, "Cannot squeeze a non-zero dim");
        v.erase(v.begin());
    }
    return graphlib::Shape::create(v);
}

bool can_be_broadcasted(const Shape &a, const Shape &b)
{
    return a.can_be_broadcasted_to(b) || b.can_be_broadcasted_to(a);
}

bool Shape::can_be_broadcasted_to(const Shape &other) const
{
    TT_ASSERT(valid_ && other.valid_);

    if (dims_.size() > other.dims_.size())
    {
        // If current shape has more dimensions than other, it cannot be broadcasted to other.
        return false;
    }

    // To be able to broadcast current shape to other shape, all dimensions of the current shape must satisfy:
    //      curr_dims[i] == other_dims[i] or curr_dims[i] == 1
    //
    // If the rank of current shape is less than the rank of the other shape, we assume that the current shape
    // dimensions are prepended with 1s to match the rank of the other shape - hence for these prepended dimensions the
    // above condition will always be satisfied.
    for (auto dim = dims_.rbegin(), other_dim = other.dims_.rbegin();
         dim != dims_.rend() && other_dim != other.dims_.rend();
         ++dim, ++other_dim)
    {
        if (*dim != *other_dim && *dim != 1)
        {
            return false;
        }
    }

    return true;
}

// Return the list of dims_ (and amount) that need to be broadcast from current to other
std::vector<DimBroadcast> Shape::broadcast_dims(const Shape &other) const
{
    std::vector<DimBroadcast> ret;
    TT_ASSERT(valid_ && other.valid_);

    if (volume() == other.volume())
    {
        // If volume is equal, we can't be bcasting, needed for reshape which isn't compatible with code below
        return {};
    }

    for (std::size_t i = 0; i < dims_.size(); i++)
    {
        if (i >= other.dims_.size())
            break;

        std::size_t my_dim = dims_[dims_.size() - 1 - i];
        std::size_t other_dim = other.dims_[other.dims_.size() - 1 - i];

        if (my_dim < other_dim)
        {
            if (type_ == FREE)
                TT_ASSERT(my_dim == 1, "Invalid broadcast shapes: " + as_string() + " vs " + other.as_string());
            else
                TT_ASSERT(
                    ((i >= 2) && my_dim == FORGE_TILE_DIM) || ((i < 2) && my_dim == 1),
                    "Invalid broadcast shapes: " + as_string() + " vs " + other.as_string());
            ret.push_back(std::make_tuple(0, dims_.size() - 1 - i, other_dim));
        }
    }
    return ret;
}

std::uint32_t Shape::rt() const
{
    TT_ASSERT(valid_, "Shape is not set.");
    TT_ASSERT(type_ == FORGE, "Accessing forge dimensions in non-forge shape.");
    int forge_tile_r = tt::graphlib::get_row_size_from_tile_size(tile_dim_);
    int dim_r = dims_.size() - 2;
    return dims_[dim_r] / forge_tile_r;
}

std::uint32_t Shape::ct() const
{
    TT_ASSERT(valid_, "Shape is not set.");
    TT_ASSERT(type_ == FORGE, "Accessing forge dimensions in non-forge shape.");
    int dim_c = dims_.size() - 1;
    return dims_[dim_c] / FORGE_TILE_DIM;
}

std::uint32_t Shape::z() const
{
    TT_ASSERT(valid_, "Shape is not set.");
    TT_ASSERT(type_ == FORGE, "Accessing forge dimensions in non-forge shape.");
    int dim_z = dims_.size() - 3;
    return dims_[dim_z];
}

std::uint32_t Shape::w() const
{
    TT_ASSERT(valid_, "Shape is not set.");
    TT_ASSERT(type_ == FORGE, "Accessing forge dimensions in non-forge shape.");
    int dim_w = dims_.size() - 4;
    return dims_[dim_w];
}

bool Shape::is_single_tile() const
{
    TT_ASSERT(type_ == FORGE, "'Single-tile' is only meaningful in forge context.");
    return w() == 1 && z() == 1 && rt() == 1 && ct() == 1;
}

bool Shape::is_unit(int index) const
{
    index = negative_index(index);
    if (is_forge() and index >= -2)
        return (*this)[index] == FORGE_TILE_DIM;
    return (*this)[index] == 1;
}

std::tuple<bool, int, std::vector<std::uint32_t>> Shape::get_pickle_data() const
{
    return std::make_tuple<>(valid_, (int)type_, dims_);
}

Shape Shape::create_from_pickled(bool valid, int type, std::vector<std::uint32_t> dims)
{
    if (!valid)
    {
        return Shape();
    }

    if (((Shape::Type)type) == Type::FREE)
    {
        return Shape::create(dims);
    }

    if (((Shape::Type)type) == Type::FORGE)
    {
        return Shape::create_forge(dims);
    }

    throw std::runtime_error("Invalid shape type");
}

}  // namespace graphlib
}  // namespace tt
