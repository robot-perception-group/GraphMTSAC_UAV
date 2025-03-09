#include "FIFOBuffer.h"

// Constructor to initialize the buffer with a maximum size
FIFOBuffer::FIFOBuffer(size_t size) : max_size(size) {}

// Method to insert a new vector
void FIFOBuffer::insert(const std::vector<float>& data) {
    if (buffer.size() == max_size) {
        buffer.pop_front(); // Remove the oldest element if the buffer is full
    }
    buffer.push_back(data); // Insert the new element
}

// Method to get the entire table
std::vector<std::vector<float>> FIFOBuffer::getTable() const {
    return std::vector<std::vector<float>>(buffer.begin(), buffer.end());
}

// Method to get the reversed table
std::vector<std::vector<float>> FIFOBuffer::getReversedTable() const {
    std::vector<std::vector<float>> reversed_table(buffer.rbegin(), buffer.rend());
    return reversed_table;
}

// Method to get the transposed table
std::vector<std::vector<float>> FIFOBuffer::getTransposedTable() const {
    if (buffer.empty()) {
        return {};
    }

    size_t num_rows = buffer.size();
    size_t num_cols = buffer[0].size();

    // Initialize the transposed table
    std::vector<std::vector<float>> transposed_table(num_cols, std::vector<float>(num_rows));

    // Fill the transposed table
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            transposed_table[j][i] = buffer[i][j];
        }
    }

    return transposed_table;
}

// Method to get the reversed and transposed table
std::vector<std::vector<float>> FIFOBuffer::getReversedTransposedTable() const {
    if (buffer.empty()) {
        return {};
    }

    size_t num_rows = buffer.size();
    size_t num_cols = buffer[0].size();

    // Initialize the transposed and reversed table
    std::vector<std::vector<float>> reversed_transposed_table(num_cols, std::vector<float>(num_rows));

    // Fill the transposed and reversed table
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            reversed_transposed_table[j][num_rows - 1 - i] = buffer[i][j];
        }
    }

    return reversed_transposed_table;
}