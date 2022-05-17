/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "myTestKernel.h"

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void myTestKernel(const int n, const float coefficient, const float* input, float* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        // output[i] = input[i] > 0 ? input[i] : input[i] * negativeSlope;
        output[i] = input[i] + coefficient;
    }
}

pluginStatus_t myTestGPU(cudaStream_t stream, const int n, const float coefficient, const void* input, void* output)
{
    const int BS = 512;
    const int GS = (n + BS - 1) / BS;
    myTestKernel<BS><<<GS, BS, 0, stream>>>(n, coefficient,
                                           (const float*) input,
                                           (float*) output);
    return STATUS_SUCCESS;
}

pluginStatus_t myTestInference(
    cudaStream_t stream, const int n, const float coefficient, const void* input, void* output)
{
    return myTestGPU(stream, n, coefficient, (const float*) input, (float*) output);
}
