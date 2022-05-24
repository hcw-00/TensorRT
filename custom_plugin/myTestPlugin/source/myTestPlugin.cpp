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
#include "myTestPlugin.h"
#include "checkMacrosPlugin.h"
#include "myTestKernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::MyTestPluginCreator;
using nvinfer1::plugin::MyTest;

static const char* MYTEST_PLUGIN_VERSION{"1"};
static const char* MYTEST_PLUGIN_NAME{"MyTest_TRT"};
PluginFieldCollection MyTestPluginCreator::mFC{};
std::vector<PluginField> MyTestPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MyTestPluginCreator);

// MyTest {{{
MyTest::MyTest(float coefficient)
    : mCoefficient(coefficient)
    , mBatchDim(1)
{
}

MyTest::MyTest(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    mCoefficient = read<float>(d);
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}

int MyTest::getNbOutputs() const noexcept
{
    return 1;
}

Dims MyTest::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

int MyTest::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = myTestInference(stream, mBatchDim * batchSize, mCoefficient, inputData, outputData);
    return status;
}

size_t MyTest::getSerializationSize() const noexcept
{
    // mCoefficient, mBatchDim
    return sizeof(float) + sizeof(int);
}

void MyTest::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mCoefficient);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

void MyTest::configureWithFormat(
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

bool MyTest::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int MyTest::initialize() noexcept
{
    return 0;
}

void MyTest::terminate() noexcept {}

size_t MyTest::getWorkspaceSize(int /* maxBatchSize */) const noexcept
{
    return 0;
}

const char* MyTest::getPluginType() const noexcept
{
    return MYTEST_PLUGIN_NAME;
}

const char* MyTest::getPluginVersion() const noexcept
{
    return MYTEST_PLUGIN_VERSION;
}

void MyTest::destroy() noexcept
{
    delete this;
}

IPluginV2* MyTest::clone() const noexcept
{
    IPluginV2* plugin = new MyTest(mCoefficient);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

MyTestPluginCreator::MyTestPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("coefficient", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MyTestPluginCreator::getPluginName() const noexcept
{
    return MYTEST_PLUGIN_NAME;
}

const char* MyTestPluginCreator::getPluginVersion() const noexcept
{
    return MYTEST_PLUGIN_VERSION;
}

const PluginFieldCollection* MyTestPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* MyTestPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    float coefficient = *(static_cast<const float*>(fields[0].data));

    return new MyTest(coefficient);
}

IPluginV2* MyTestPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call LReluPlugin::destroy()
    return new MyTest(serialData, serialLength);
}
// MyTest }}}
