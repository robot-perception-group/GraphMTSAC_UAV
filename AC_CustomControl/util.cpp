#include "util.h"
#include <cassert>
#include <cmath>
#include <algorithm>

// ------------------------------------------------------------------------
// Existing utility functions
// ------------------------------------------------------------------------

// for printing a 1D std::vector<float>
// #include <sstream>
// #include <iostream>
// std::string vectorToString(const std::vector<float>& vec) {
//     std::string result = "[";
//     for (size_t i = 0; i < vec.size(); ++i) {
//         result += std::to_string(vec[i]);
//         if (i != vec.size() - 1) {
//             result += ", ";
//         }
//     }
//     result += "]";
//     return result;
// }

// for mapping angle to [-pi, pi]
float mapAngleToRange(float angle) {
    // Normalize the angle to the range [-pi, pi]
    const float PI = 3.14159265358979323846f;
    angle = fmod(angle + PI, 2.0f * PI);
    if (angle < 0) {
        angle += 2.0f * PI;
    }
    return angle - PI;
}

// get the last column of a 2D vector
std::vector<float> getLastColumn(const std::vector<std::vector<float>>& matrix) {
    std::vector<float> last_column;
    for (const auto& row : matrix) {
        if (!row.empty()) {
            last_column.push_back(row.back());
        }
    }
    return last_column;
}

// clamp elements to range [min_val, max_val]
void clampToRange(std::vector<float>& vec, float min_val, float max_val) {
    for (float& element : vec) {
        element = std::max(min_val, std::min(element, max_val));
    }
}

// concatenate two vectors
std::vector<float> vecCat(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    std::vector<float> result = vec1; 
    result.insert(result.end(), vec2.begin(), vec2.end());
    return result;
}

// vector-wise addition
std::vector<float> vecAdd(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    assert(vec1.size() == vec2.size());
    std::vector<float> result(vec1.size(), 0.0f);
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}

// 2D vector-wise addition
std::vector<std::vector<float>> vec2DAdd(const std::vector<std::vector<float>>& c1,
                                         const std::vector<std::vector<float>>& buf) {
    // Ensure dimensions match
    assert(!c1.empty() && !buf.empty());
    assert(c1.size() == buf.size() && c1[0].size() == buf[0].size());

    std::vector<std::vector<float>> result(c1.size(), std::vector<float>(c1[0].size(), 0.0f));
    for (size_t i = 0; i < c1.size(); ++i) {
        for (size_t j = 0; j < c1[i].size(); ++j) {
            result[i][j] = c1[i][j] + buf[i][j];
        }
    }
    return result;
}

// ReLU for 1D
std::vector<float> relu(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::max(0.0f, x[i]);
    }
    return result;
}

// ReLU for 2D
std::vector<std::vector<float>> relu2D(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> result = input; 
    for (auto& row : result) {
        for (auto& element : row) {
            element = std::max(0.0f, element);
        }
    }
    return result;
}

// softmax for 1D
std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> exp_x(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum_exp = 0.0f;

    for (size_t i = 0; i < x.size(); ++i) {
        exp_x[i] = std::exp(x[i] - max_val);
        sum_exp += exp_x[i];
    }
    for (size_t i = 0; i < x.size(); ++i) {
        exp_x[i] /= sum_exp;
    }
    return exp_x;
}

// chomp the last "padding" columns from a 2D vector
std::vector<std::vector<float>> chomp1d(const std::vector<std::vector<float>>& input, int padding) {
    std::vector<std::vector<float>> result;
    for (const auto& row : input) {
        std::vector<float> new_row(row.begin(), row.end() - padding);
        result.push_back(new_row);
    }
    return result;
}

// apply zero-padding on each row
std::vector<std::vector<float>> applyPadding(const std::vector<std::vector<float>>& input, int padding) {
    int C_in = static_cast<int>(input.size());
    int L_in = static_cast<int>(input[0].size());
    int padded_L_in = L_in + 2 * padding;

    std::vector<std::vector<float>> padded_input(C_in, std::vector<float>(padded_L_in, 0.0f));
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int l_in = 0; l_in < L_in; ++l_in) {
            padded_input[c_in][l_in + padding] = input[c_in][l_in];
        }
    }
    return padded_input;
}

// 1D conv
std::vector<std::vector<float>> conv1d(
    const std::vector<std::vector<float>>& input,
    const std::vector<std::vector<std::vector<float>>>& weight,
    const std::vector<float>& bias,
    int stride, int padding, int dilation) 
{
    int C_in = static_cast<int>(input.size());
    int L_in = static_cast<int>(input[0].size());
    int C_out = static_cast<int>(weight.size());
    int K = static_cast<int>(weight[0][0].size());

    // output length
    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    // Initialize output
    std::vector<std::vector<float>> output(C_out, std::vector<float>(L_out, 0.0f));
    // padded input
    std::vector<std::vector<float>> input_padded = applyPadding(input, padding);

    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int l_out = 0; l_out < L_out; ++l_out) {
            for (int c_in = 0; c_in < C_in; ++c_in) {
                for (int k = 0; k < K; ++k) {
                    int l_in = l_out * stride + k * dilation;
                    output[c_out][l_out] += input_padded[c_in][l_in] * weight[c_out][c_in][k];
                }
            }
            if (!bias.empty()) {
                output[c_out][l_out] += bias[c_out];
            }
        }
    }
    return output;
}

// dot product
float vecDot(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    assert(vec1.size() == vec2.size());
    float result = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// matrix-vector multiply
std::vector<float> matVecMul(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec) {
    assert(!mat.empty());
    assert(mat[0].size() == vec.size());
    std::vector<float> result(mat.size(), 0.0f);
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

// linear layer + optional ReLU
std::vector<float> linear_layer(const std::vector<float>& bias,
                                const std::vector<std::vector<float>>& weight,
                                const std::vector<float>& input,
                                bool activation) {
    std::vector<float> output = vecAdd(bias, matVecMul(weight, input));
    if (activation) {
        return relu(output);
    }
    return output;
}

// composition layer
std::vector<float> composition_layer(
    const std::vector<std::vector<std::vector<float>>>& weight, // [contextdim][outdim][indim]
    const std::vector<std::vector<float>>& bias,               // [contextdim][outdim]
    const std::vector<float>& comp_weight,                     // [contextdim]
    const std::vector<float>& input,                           // [indim]
    bool activation)
{
    // dims
    size_t contextdim = weight.size();
    size_t outdim     = weight[0].size();
    bool has_bias     = !bias.empty();

    // For each context i: x[i] = bias[i] + weight[i]*input
    std::vector<std::vector<float>> x(contextdim, std::vector<float>(outdim, 0.0f));
    for (size_t i = 0; i < contextdim; ++i) {
        if (has_bias) {
            x[i] = vecAdd(bias[i], matVecMul(weight[i], input));
        } else {
            x[i] = matVecMul(weight[i], input);
        }
    }

    // output[j] = sum_i( comp_weight[i] * x[i][j] ), j in [0..outdim-1]
    std::vector<float> output(outdim, 0.0f);
    for (size_t i = 0; i < contextdim; ++i) {
        for (size_t j = 0; j < outdim; ++j) {
            output[j] += comp_weight[i] * x[i][j];
        }
    }
    if (activation) {
        return relu(output);
    } else {
        return output;
    }
}

// ------------------------------------------------------------------------
// GraphNN-style inline helpers (merged in the same file)
// ------------------------------------------------------------------------
namespace GraphNN {

// matmul_1d:  out_dim x in_dim times in_dim
std::vector<float> matmul_1d(
    const std::vector<float> &in,    
    const std::vector<float> &W,     
    int out_dim, int in_dim)
{
    // out = W * in
    std::vector<float> out(out_dim, 0.0f);
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            sum += W[i * in_dim + j] * in[j];
        }
        out[i] = sum;
    }
    return out;
}

std::vector<float> vec_add(const std::vector<float> &a,
                           const std::vector<float> &b)
{
    assert(a.size() == b.size());
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        out[i] = a[i] + b[i];
    }
    return out;
}

void relu_inplace(std::vector<float> &x) {
    for (auto &val : x) {
        if (val < 0.0f) {
            val = 0.0f;
        }
    }
}

std::vector<float> softmax(const std::vector<float> &x) {
    float max_val = *std::max_element(x.begin(), x.end());
    float sum_exp = 0.0f;
    for (float v : x) {
        sum_exp += std::exp(v - max_val);
    }
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        out[i] = std::exp(x[i] - max_val) / sum_exp;
    }
    return out;
}

std::vector<float> linear_layer_1d(
    const std::vector<float> &W,     
    const std::vector<float> &b,     
    const std::vector<float> &in,    
    int out_dim, int in_dim,
    bool do_relu, bool do_softmax)
{
    auto out = matmul_1d(in, W, out_dim, in_dim);
    out = vec_add(out, b);
    if (do_relu) {
        relu_inplace(out);
    }
    if (do_softmax) {
        out = softmax(out);
    }
    return out;
}

inline float meanOf(const std::vector<float> &v) {
    float sum = 0.0f;
    for (auto &x : v) sum += x;
    return sum / (float)v.size();
}

inline float varOf(const std::vector<float> &v, float mean) {
    float sumSq = 0.0f;
    for (auto &x : v) {
        float diff = x - mean;
        sumSq += diff*diff;
    }
    return sumSq / (float)v.size();
}

void layernorm_inplace(std::vector<float> &x,
                       const std::vector<float> &gamma,
                       const std::vector<float> &beta,
                       float eps)
{
    float m = meanOf(x);
    float v = varOf(x, m);
    for (size_t i = 0; i < x.size(); i++) {
        float normed = (x[i] - m) / std::sqrt(v + eps);
        x[i] = normed * gamma[i] + beta[i];
    }
}

std::vector<float> parallel_fc_1batch(
    const std::vector<float> &inputs,  
    int n_parallels,
    int in_dim,
    int out_dim,
    const std::vector<float> &weight,  
    const std::vector<float> &bias,    
    bool relu)
{
    std::vector<float> out(n_parallels * out_dim, 0.0f);
    // weight is shape [n_parallels, out_dim, in_dim], flattened row-major
    for (int i = 0; i < n_parallels; i++) {
        for (int od = 0; od < out_dim; od++) {
            float sum = 0.0f;
            for (int id = 0; id < in_dim; id++) {
                float w_ij = weight[i * (out_dim*in_dim) + od*in_dim + id];
                sum += w_ij * inputs[i * in_dim + id];
            }
            float b_ij = bias[i * out_dim + od];
            float val = sum + b_ij;
            if (relu && val < 0.0f) val = 0.0f;
            out[i * out_dim + od] = val;
        }
    }
    return out;
}

// GCN for a single batch
std::vector<float> gcn_1batch(
    const std::vector<float> &H,  
    int N_nodes,
    int N_action_nodes,
    int in_dim,
    const std::vector<float> &A,  
    const std::vector<float> &W,  
    const std::vector<float> &b,  
    int out_dim,
    bool use_layernorm,
    const std::vector<float> &ln_weight,
    const std::vector<float> &ln_bias,
    float eps)
{
    // 1) out_matmul = A @ H
    std::vector<float> out_matmul(N_nodes * in_dim, 0.0f);
    for (int i = 0; i < N_nodes; i++) {
        for (int j = 0; j < N_nodes; j++) {
            float a_ij = 0;
            if (i<N_action_nodes){  // only compute last 3 rows
                a_ij = A[i*N_nodes + j];
            }

            for (int d = 0; d < in_dim; d++) {
                out_matmul[i*in_dim + d] += a_ij * H[j*in_dim + d];
            }
        }
    }
    // 2) out_lin = out_matmul @ W^T + b
    std::vector<float> out_lin(N_nodes * out_dim, 0.0f);
    for (int i = 0; i < N_nodes; i++) {
        for (int od = 0; od < out_dim; od++) {
            float sum = 0.0f;
            for (int d = 0; d < in_dim; d++) {
                sum += W[od*in_dim + d] * out_matmul[i*in_dim + d];
            }
            sum += b[od];
            out_lin[i*out_dim + od] = sum;
        }
    }
    // 3) optional LN + ReLU
    for (int i = 0; i < N_nodes; i++) {
        // shape [out_dim]
        std::vector<float> rowVec(out_dim);
        for (int od = 0; od < out_dim; od++) {
            rowVec[od] = out_lin[i*out_dim + od];
        }
        if (use_layernorm && ln_weight.size() == (size_t)out_dim) {
            float m = meanOf(rowVec);
            float v = varOf(rowVec, m);
            for (int od = 0; od < out_dim; od++) {
                float normed = (rowVec[od] - m) / std::sqrt(v + eps);
                normed = normed * ln_weight[od] + ln_bias[od];
                rowVec[od] = normed;
            }
        }
        // ReLU
        for (int od = 0; od < out_dim; od++) {
            if (rowVec[od] < 0.0f) rowVec[od] = 0.0f;
            out_lin[i*out_dim + od] = rowVec[od];
        }
    }
    return out_lin;
}


} // namespace GraphNN

// ------------------------------------------------------------------------
// End of util.cpp
// ------------------------------------------------------------------------
