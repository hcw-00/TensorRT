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
#include "lReluTestPlugin.h"
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::LReluTestPluginCreator;
using nvinfer1::plugin::LReLUTest;

static const char* LRELUTEST_PLUGIN_VERSION{"1"};
static const char* LRELUTEST_PLUGIN_NAME{"LReLUTest_TRT"};
PluginFieldCollection LReluTestPluginCreator::mFC{};
std::vector<PluginField> LReluTestPluginCreator::mPluginAttributes;

// LeakyReLU {{{
LReLUTest::LReLUTest(float negSlope)
    : mNegSlope(negSlope)
    , mBatchDim(1)
{
}

LReLUTest::LReLUTest(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    mNegSlope = read<float>(d);
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}

int LReLUTest::getNbOutputs() const noexcept
{
    return 1;
}

Dims LReLUTest::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

int LReLUTest::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = lReLUTestInference(stream, mBatchDim * batchSize, mNegSlope, inputData, outputData);
    return status;
}

size_t LReLUTest::getSerializationSize() const noexcept
{
    // mNegSlope, mBatchDim
    return sizeof(float) + sizeof(int);
}

void LReLUTest::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

void LReLUTest::configureWithFormat(
    const Dims* inputDims, int /* nbInputs */, const Dims* /* outputDims */, int nbOutputs, DataType type, PluginFormat format, int) noexcept
{
    ASSERT(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    ASSERT(mBatchDim == 1);
    ASSERT(nbOutputs == 1);
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool LReLUTest::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int LReLUTest::initialize() noexcept
{
    return 0;
}

void LReLUTest::terminate() noexcept {}

size_t LReLUTest::getWorkspaceSize(int /* maxBatchSize */) const noexcept
{
    return 0;
}

const char* LReLUTest::getPluginType() const noexcept
{
    return LRELUTEST_PLUGIN_NAME;
}

const char* LReLUTest::getPluginVersion() const noexcept
{
    return LRELUTEST_PLUGIN_VERSION;
}

void LReLUTest::destroy() noexcept
{
    delete this;
}

IPluginV2* LReLUTest::clone() const noexcept
{
    IPluginV2* plugin = new LReLUTest(mNegSlope);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

LReluTestPluginCreator::LReluTestPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LReluTestPluginCreator::getPluginName() const noexcept
{
    return LRELUTEST_PLUGIN_NAME;
}

const char* LReluTestPluginCreator::getPluginVersion() const noexcept
{
    return LRELUTEST_PLUGIN_VERSION;
}

const PluginFieldCollection* LReluTestPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LReluTestPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    float negSlope = *(static_cast<const float*>(fields[0].data));

    return new LReLUTest(negSlope);
}

IPluginV2* LReluTestPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call LReluPlugin::destroy()
    return new LReLUTest(serialData, serialLength);
}
// LeakReLU }}}
