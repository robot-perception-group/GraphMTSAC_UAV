#pragma once

#ifndef UTIL_H
#define UTIL_H

#include <vector>

// -------------------------------------------------
// Existing utility function declarations
// -------------------------------------------------

// for printing
// #include <string>
// std::string vectorToString(const std::vector<float>& vec);

// map angle to [-pi, pi]
float mapAngleToRange(float angle);

// get last column of a 2D vector
std::vector<float> getLastColumn(const std::vector<std::vector<float>>& matrix);

// clamp elements to [min_val, max_val]
void clampToRange(std::vector<float>& vec, float min_val, float max_val);

// concatenate two vectors
std::vector<float> vecCat(const std::vector<float>& vec1, const std::vector<float>& vec2);

// vector add
std::vector<float> vecAdd(const std::vector<float>& vec1, const std::vector<float>& vec2);

// 2D vector add
std::vector<std::vector<float>> vec2DAdd(const std::vector<std::vector<float>>& c1,
                                         const std::vector<std::vector<float>>& buf);

// ReLU 1D
std::vector<float> relu(const std::vector<float>& x);

// ReLU 2D
std::vector<std::vector<float>> relu2D(const std::vector<std::vector<float>>& input);

// softmax 1D
std::vector<float> softmax(const std::vector<float>& x);

// remove last `padding` columns
std::vector<std::vector<float>> chomp1d(const std::vector<std::vector<float>>& input, int padding);

// apply zero-padding around 1D signals
std::vector<std::vector<float>> applyPadding(const std::vector<std::vector<float>>& input, int padding);

// 1D convolution
std::vector<std::vector<float>> conv1d(
    const std::vector<std::vector<float>>& input,
    const std::vector<std::vector<std::vector<float>>>& weight,
    const std::vector<float>& bias,
    int stride, int padding, int dilation);

// dot product
float vecDot(const std::vector<float>& vec1, const std::vector<float>& vec2);

// matrix-vector multiply
std::vector<float> matVecMul(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec);

// basic linear layer
std::vector<float> linear_layer(
    const std::vector<float>& bias,
    const std::vector<std::vector<float>>& weight,
    const std::vector<float>& input,
    bool activation);

// "composition_layer" logic
std::vector<float> composition_layer(
    const std::vector<std::vector<std::vector<float>>>& weight, // [contextdim, outdim, indim]
    const std::vector<std::vector<float>>& bias,               // [contextdim, outdim]
    const std::vector<float>& comp_weight,                     // [contextdim]
    const std::vector<float>& input,                           // [indim]
    bool activation);

// -------------------------------------------------
// GraphNN namespace for GCN / ParallelFC logic
// -------------------------------------------------
namespace GraphNN {

    // matmul for (out_dim x in_dim) * (in_dim)
    std::vector<float> matmul_1d(
        const std::vector<float> &in,
        const std::vector<float> &W,
        int out_dim, int in_dim
    );

    // simple vector add
    std::vector<float> vec_add(const std::vector<float> &a, const std::vector<float> &b);

    // in-place ReLU
    void relu_inplace(std::vector<float> &x);

    // softmax
    std::vector<float> softmax(const std::vector<float> &x);

    // linear layer with optional ReLU/softmax
    std::vector<float> linear_layer_1d(
        const std::vector<float> &W, 
        const std::vector<float> &b, 
        const std::vector<float> &in,
        int out_dim,
        int in_dim,
        bool do_relu,
        bool do_softmax
    );

    // layernorm in-place
    void layernorm_inplace(
        std::vector<float> &x,
        const std::vector<float> &gamma,
        const std::vector<float> &beta,
        float eps = 1e-5f
    );

    // ParallelFC for a single batch
    std::vector<float> parallel_fc_1batch(
        const std::vector<float> &inputs,  // shape [n_parallels * in_dim]
        int n_parallels,
        int in_dim,
        int out_dim,
        const std::vector<float> &weight,  // shape [n_parallels, out_dim, in_dim] flattened
        const std::vector<float> &bias,    // shape [n_parallels, out_dim]
        bool relu
    );

    // GCN for a single batch
    std::vector<float> gcn_1batch(
        const std::vector<float> &H,  // [N_nodes * in_dim]
        int N_nodes,
        int N_action_nodes,
        int in_dim,
        const std::vector<float> &A,  // adjacency [N_nodes * N_nodes]
        const std::vector<float> &W,  // [out_dim * in_dim]
        const std::vector<float> &b,  // [out_dim]
        int out_dim,
        bool use_layernorm,
        const std::vector<float> &ln_weight,
        const std::vector<float> &ln_bias,
        float eps = 1e-5f
    );

} // namespace GraphNN

#endif // UTIL_H
