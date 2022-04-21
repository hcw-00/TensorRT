#include "kernel.h"

pluginStatus_t testInference(
    cudaStream_t stream, const int n, const float coefficient, const void* input, void* output);

