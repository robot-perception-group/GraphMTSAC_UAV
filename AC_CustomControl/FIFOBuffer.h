#ifndef FIFOBUFFER_H
#define FIFOBUFFER_H

#include <vector>
#include <deque>

class FIFOBuffer {
private:
    std::deque<std::vector<float>> buffer;
    size_t max_size;

public:
    // Constructor to initialize the buffer with a maximum size
    FIFOBuffer(size_t size);

    // Method to insert a new vector
    void insert(const std::vector<float>& data);

    // Method to get the entire table
    std::vector<std::vector<float>> getTable() const;

    // Method to get the reversed table
    std::vector<std::vector<float>> getReversedTable() const;

    std::vector<std::vector<float>> getTransposedTable() const;
    
    std::vector<std::vector<float>> getReversedTransposedTable() const;
};

#endif // FIFOBUFFER_H