#! /bin/bash

# Export retinanet
python3 /workspace/TensorRT/export/build_engine.py --onnx $1 --engine $2 --precision $3 --calib_batch_size $4