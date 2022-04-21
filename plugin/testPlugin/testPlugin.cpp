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
#include "checkMacrosPlugin.h"
#include "kernel.h"
#include "testPlugin.h"
#include "testPluginKernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::TestPluginCreator;
using nvinfer1::plugin::TEST;

static const char* TEST_PLUGIN_VERSION{"1"};
static const char* TEST_PLUGIN_NAME{"TEST_TRT"};
PluginFieldCollection TestPluginCreator::mFC{};
std::vector<PluginField> TestPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(TestPluginCreator);

TEST::TEST(float coefficient)
    : mCoefficient(coefficient)
    , mBatchDim(1)
{
}

TEST::TEST(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    mCoefficient = read<float>(d);
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}

int TEST::getNbOutputs() const noexcept
{
    return 1;
}

Dims TEST::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

int TEST::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = testInference(stream, mBatchDim * batchSize, mCoefficient, inputData, outputData);
    return status;
}

size_t TEST::getSerializationSize() const noexcept
{
    // mCoefficient, mBatchDim
    return sizeof(float) + sizeof(int);
}

void TEST::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mCoefficient);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

void TEST::configureWithFormat(
    const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int) noexcept
{
    ASSERT(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    ASSERT(mBatchDim == 1);
    ASSERT(nbOutputs == 1);
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool TEST::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int TEST::initialize() noexcept
{
    return 0;
}

void TEST::terminate() noexcept {}

size_t TEST::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return 0;
}

const char* TEST::getPluginType() const noexcept
{
    return TEST_PLUGIN_NAME;
}

const char* TEST::getPluginVersion() const noexcept
{
    return TEST_PLUGIN_VERSION;
}

void TEST::destroy() noexcept
{
    delete this;
}

IPluginV2* TEST::clone() const noexcept
{
    IPluginV2* plugin = new TEST(mCoefficient);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

TestPluginCreator::TestPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("coefficient", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* TestPluginCreator::getPluginName() const noexcept
{
    return TEST_PLUGIN_NAME;
}

const char* TestPluginCreator::getPluginVersion() const noexcept
{
    return TEST_PLUGIN_VERSION;
}

const PluginFieldCollection* TestPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* TestPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    float coefficient = *(static_cast<const float*>(fields[0].data));

    return new TEST(coefficient);
}

IPluginV2* TestPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call testPlugin::destroy()
    return new TEST(serialData, serialLength);
}