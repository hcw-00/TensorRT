from email.policy import default
import os
import sys
import argparse
import logging

# import tensorflow as tf
import onnx_graphsurgeon as gs
import numpy as np
import onnx
# import onnx_utils

def add_efficient_nms_plugin(args):
    onnx_path = args.onnx_path
    graph = gs.import_onnx(onnx.load(onnx_path))
    save_name = os.path.splitext(onnx_path)[0] + \
        f'_plugin.onnx'
    tmap = graph.tensors()
    x, w, y = tmap["X"], tmap["W"], tmap["Y"]
    plugin_inputs = [y]

    plugin_op = "LReLUTest_TRT"
    import pdb;pdb.set_trace()
    outputs = gs.Variable(name="outputs", dtype=np.float32, shape=[1, 5, 222, 222])
    node = gs.Node(op=plugin_op, inputs=[y], outputs=[outputs])
    graph.nodes.append(node)
    
    # # Exporters
    onnx.save(gs.export_onnx(graph), save_name)


def main(args):
    add_efficient_nms_plugin(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", required=True, help="The input ONNX model file to load")
    # parser.add_argument("--num_detections", type=int, default=2000)
    # parser.add_argument("--score_threshold", type=float, default=0.2)
    # parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)