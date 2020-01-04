!alias python=python3
!alias pip=pip3
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from keras.models import load_model, Model
from keras import backend as K
import tensorflow as tf
from utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.6):


 box_scores = box_confidence*box_class_probs
 box_classes = K.argmax(box_scores,axis = -1)
 box_class_scores = K.max(box_scores,axis = -1,keepdims=False)

 filtering_mask = (box_class_scores >= threshold)

 scores = tf.boolean_mask(box_class_scores, filtering_mask)
 boxes = tf.boolean_mask(boxes, filtering_mask)
 classes = tf.boolean_mask(box_classes, filtering_mask)

 return scores, boxes, classes

with tf.Session() as test_a:
 box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 51)
 boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 51)
 box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 51)
 scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, thre
 print("scores[2] = " + str(scores[2].eval()))
 print("boxes[2] = " + str(boxes[2].eval()))
 print("classes[2] = " + str(classes[2].eval()))
 print("scores.shape = " + str(scores.shape))
 print("boxes.shape = " + str(boxes.shape))
 print("classes.shape = " + str(classes.shape))

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):


 max_boxes_tensor = K.variable(max_boxes, dtype='int32') # tensor to be used in tf.ima

 K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

 nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold) ##
 ####Use tf.gather() to pick boxes and their respective predicted classes as well as the
 scores = K.gather(scores,nms_indices)
 boxes = K.gather(boxes,nms_indices)
 classes = K.gather(classes,nms_indices)

 return scores, boxes, classes

with tf.Session() as test_b:
 scores = tf.random_normal([54,], mean=1, stddev=4, seed = 51)
 boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 51)
 classes = tf.random_normal([54,], mean=1, stddev=4, seed = 51)
 scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
 init=tf.global_variables_initializer()
 test_b.run(init)
 print("scores[2] = " + str(scores[2].eval()))
 print("boxes[2] = " + str(boxes[2].eval()))
 print("classes[2] = " + str(classes[2].eval()))
 print("scores.shape = " + str(scores.eval().shape))
 print("boxes.shape = " + str(boxes.eval().shape))
 print("classes.shape = " + str(classes.eval().shape))

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=5, score_threshold=.6, i

 box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
 boxes = yolo_boxes_to_corners(box_xy, box_wh)
 scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, thre
 ###call yolo_filter_boxes to extract box confidence scores

 ### the scale_box function adjest the box dimensions according to the image.
 boxes = scale_boxes(boxes, image_shape)
 scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = 1
 ### call yolo_non_max_suppression to eliminate overlapping boxes
 return scores, boxes, classes
with tf.Session() as test_b:
 yolo_outputs = (tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 51),
 tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 51),
 tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 51),
 tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 51))
 scores, boxes, classes = yolo_eval(yolo_outputs)
 init=tf.global_variables_initializer()
 test_b.run(init)
 print("scores[2] = " + str(scores[2].eval()))
 print("boxes[2] = " + str(boxes[2].eval()))
 print("classes[2] = " + str(classes[2].eval()))
 print("scores.shape = " + str(scores.eval().shape))
 print("boxes.shape = " + str(boxes.eval().shape))
 print("classes.shape = " + str(classes.eval().shape))
sess = K.get_session()
class_names = read_classes("model_data/pascal_classes.txt")
anchors = read_anchors("model_data/yolo_tiny_anchors.txt")
image_shape = (375., 500.)
yolo_model = load_model("model_data/yolo_tiny.h5")
yolo_model.summary()

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
def predict(sess, image_file):

 image, image_data = preprocess_image(image_file, model_image_size = (416, 416))
 out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo


 print('Found {} boxes for {}'.format(len(out_boxes), image_file))
 colors = generate_colors(class_names)

 draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

 image.save(os.path.join("out", image_file), quality=90)

 output_image = scipy.misc.imread(os.path.join("out", image_file))
 imshow(output_image)

 return out_scores, out_boxes, out_classes

predict(sess, "men.JPEG")
