from email.policy import default
import os
import sys
import argparse
import logging

# import tensorflow as tf
import onnx_graphsurgeon as gs
import numpy as np
import onnx
import onnx_utils

def add_efficient_nms_plugin(args):
    onnx_path = args.onnx_path
    save_name = os.path.splitext(onnx_path)[0] + \
        f'_gs_score_threshold_{args.score_threshold}_iou_threshold_{args.iou_threshold}_max_output_boxes_{args.num_detections}.onnx'
    num_detections = args.num_detections
    # Importers
    graph = gs.import_onnx(onnx.load(onnx_path))

    # define input
    #import pdb;pdb.set_trace()
    box_net_tensor = graph.tensors()['box_delta']
    pred_scores_tensor = graph.tensors()['pred_logits']
    anchors_tensor = graph.tensors()['anchors']
    nms_inputs = [box_net_tensor, pred_scores_tensor, anchors_tensor]

    nms_op = "EfficientNMS_TRT"
    nms_attrs = {
        'plugin_version': "1",
        'background_class': -1,
        'max_output_boxes': num_detections,
        'score_threshold': max(0.01, args.score_threshold),  # Keep threshold to at least 0.01 for better efficiency
        'iou_threshold': args.iou_threshold,
        'score_activation': True,
        'box_coding': 1
    }
    nms_output_classes_dtype = np.int32
    # NMS Outputs
    batch_size = 1
    nms_output_num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=[batch_size, 1])
    nms_output_boxes = gs.Variable(name="detection_boxes", dtype=np.float32, shape=[batch_size, num_detections, 4])
    nms_output_scores = gs.Variable(name="detection_scores", dtype=np.float32, shape=[batch_size, num_detections])
    nms_output_classes = gs.Variable(name="detection_classes", dtype=nms_output_classes_dtype, shape=[batch_size, num_detections])

    nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]

    graph.plugin(
        op=nms_op,
        name="nms/non_maximum_suppression",
        inputs=nms_inputs,
        outputs=nms_outputs,
        attrs=nms_attrs)

    graph.outputs = nms_outputs

    graph.cleanup().toposort()
    # Exporters
    onnx.save(gs.export_onnx(graph), save_name)


def main(args):
    add_efficient_nms_plugin(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", required=True, help="The input ONNX model file to load")
    parser.add_argument("--num_detections", type=int, default=2000)
    parser.add_argument("--score_threshold", type=float, default=0.2)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)