import ctypes
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch

LIB_PATH = "/workspace/TensorRT/testplugin/leakyReluTestPlugin/build/liblrelutest_trt.so"
ctypes.CDLL(LIB_PATH)
TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

def get_trt_plugin(plugin_name):
        plugin = None
        for plugin_creator in PLUGIN_CREATORS:
            if plugin_creator.name == plugin_name:
                coefficient_field = trt.PluginField("coefficient", np.array([2], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                field_collection = trt.PluginFieldCollection([coefficient_field])
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
        return plugin
tgt_plugin = get_trt_plugin("LReLUTest_TRT")
# import pdb;pdb.set_trace()
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
# network = builder.create_network(1 <<
#        int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

config = builder.create_builder_config()
config.max_workspace_size = 2**20

input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=(1, 1, 1))
test_plugin = network.add_plugin_v2(inputs=[input_layer], plugin=tgt_plugin)
test_plugin.get_output(0).name = "outputs"
network.mark_output(tensor=test_plugin.get_output(0))


serialized_engine = builder.build_serialized_network(network, config)

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

# print(engine.num_bindings)

allocations = []
inputs = []
outputs = []
for i in range(engine.num_bindings):
    is_input = False
    if engine.binding_is_input(i):
        is_input = True
    print(engine.get_binding_name(i))
    print(engine.get_binding_dtype(i))
    print(engine.get_binding_shape(i))
    # import pdb;pdb.set_trace()
    name = engine.get_binding_name(i)
    dtype = engine.get_binding_dtype(i)
    shape = engine.get_binding_shape(i)

    size = np.dtype(trt.nptype(dtype)).itemsize
    for s in shape:
        size *=s
    
    batch_size=shape[0]
    allocation = cuda.mem_alloc(size)

    binding = {
        'index': i,
        'name': name,
        'dtype': np.dtype(trt.nptype(dtype)),
        'shape': list(shape),
        'allocation': allocation,
    }

    if is_input:
        allocations.append(allocation)
        inputs.append(binding)
    else:
        allocations.append(allocation)
        outputs.append(binding)

batch = torch.Tensor([[[.1]]])
print(batch.shape)
cuda.memcpy_htod(inputs[0]['allocation'], np.ascontiguousarray(batch))

context.execute_v2(allocations)
outs = [o['allocation'] for o in outputs]

print(outs)