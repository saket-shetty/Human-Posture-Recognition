import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from tkinter import *
import tkinter.filedialog
from PIL import Image
from PIL import ImageTk

from utils import label_map_util
from utils import visualization_utils as vis_util


def select_image():

    path = tkinter.filedialog.askopenfilename()

    MODEL_NAME = 'inference_graph'
    IMAGE_NAME = 'fullbody3.jpg'

    CWD_PATH = os.getcwd()


    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

    NUM_CLASSES = 3

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image = cv2.imread(path)
    image = cv2.resize(image, (600,500), None , 500, 500)

    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})



    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=False,
        max_boxes_to_draw=1,
        line_thickness=5,
        min_score_thresh=0.50)

    cv2.imshow('Object detector', image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()




root = Tk()

root.option_add("*Button.Background", "#03dac5")
root.option_add("*Button.Foreground", "#1f618d")

root.title('Human Posture Recognition')

root.geometry("600x300") 
root.resizable(0, 0)

back = Frame(master=root,bg='#6200ee')
back.pack_propagate(0)
back.pack(fill=BOTH, expand=1)

info = Label(master=back, text='Human Posture Recognition System', bg='#6200ee', fg='white',font='Verdana 22 bold')
info.pack(padx=2,pady=10,side=TOP)
go = Button(master=back, text='Choose Image',height=2,width=15,relief=RAISED, command=select_image)
go.pack(padx=2,pady=20,side=TOP)
close = Button(master=back, text='Quit',height=2,width=15, command=root.destroy)
close.pack(padx=2,pady=20,side=TOP)

root.mainloop()