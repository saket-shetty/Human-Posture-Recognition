# Human-Posture-Recognition
Human Posture Recognition using deep learning 

### Demo

[![Link to my YouTube video!](https://github.com/saket-shetty/Human-Posture-Recognition/blob/master/Capture.PNG)](https://www.youtube.com/watch?v=y_ok2L09Sq0)

### Prerequisites

1. Install Anaconda
2. Clone this repo and paste it in a folder called tenforflow1 and paste it in C: directory
3. Clone [labelImg repo](https://github.com/tzutalin/labelImg) and paste it in desktop
4. Goto models\research\object_detection\images folder remove all the images from train and test folder along with .xml file and delete both .csv files

### Actual Work

### 1. Open Anaconda and create a conda

```
conda create -n tensorflow1
```
### 2. Then activate that conda:

```
activate tensorflow1
```
### 3. Install following pip

```
(tensorflow1) C:\> pip install tensorflow
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install numpy=1.16.4 (this specific version is required since updated version is giving some error)
(tensorflow1) C:\> pip install contextlib2
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python

```
### 4.0 Enter the follow steps (all this steps should be done in tensorflow1 in anaconda virtual environment)

### 4.1 Enter the following
```
a) set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
b) set PATH=%PATH%;PYTHONPATH
```
you have to do this steps evertime when you want to train new data

### 4.2 Change directory to C:\tensorflow1\models\research then paste following
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
### 4.3 Then run the setup file
```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```
To verify whether all the setup is done correctly run following command
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
It should open a new browser mainly Internet explorer it will take some time to run in step 4 and 5 it will take few minutes since it has to download few file.
Once all the steps are correctly done two picture will be shown one with dog and another with beaches and kites once this is done correctly we can move on to next steps.

Note: If you run the full Jupyter Notebook without getting any errors, but the labeled pictures still don't appear,
try this: go in to object_detection/utils/visualization_utils.py
and comment out the import statements around lines 29 and 30 that include matplotlib. Then, try re-running the Jupyter notebook.

### 5 Gather and Label Pictures
Now we can do the actual training of the images

### 5.1 Download all the images with different angles to which you have to train the system and put it in train and test folder (C:\tensorflow1\models\research\object_detection\images)
put 80% of the images in test folder and 20% in train folder.

### 5.2 Label Image (hope you have cloned the labelImg repo)

Open command Prompt and type following
```
cd Desktop\labelImage
python labelImg.py
```
It will open a software click on open directory and link to the test and training folder (C:\tensorflow1\models\research\object_detection\images\train or test)
It should show you train images select the object from the image after that a textfield will appear enter the name of the object, do similar to the test images
after that same amount of xml file would appear in the folder as the amount of the images present.

### 5.3 change xml to CSV file
Run following command for it

```
(tensorflow1) C:\tensorflow1\models\research\object_detection>python xml_to_csv.py
```
It will create to .csv file naming test_label.csv and train_label.csv

### 5.4 <br>
Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt
For example, say you are training a classifier to detect basketballs, shirts, and shoes. You will replace the following code in generate_tfrecord.py:
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'standing':
        return 1
    elif row_label == 'sitting':
        return 2
    elif row_label == 'sleeping':
        return 3
    else:
        None
```
to your label which can be 
```

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'football':
        return 2
    elif row_label == 'sleeping':
        return 3
    else:
        None
```
### 5.5 Then, generate the TFRecord files by issuing these commands from the \object_detection folder:
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

### 6.0 Few steps before training

The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder. (Make sure the file type is .pbtxt, not .txt !) In the text editor, copy or type in the label map in the format below (the example below is the label map for my Pinochle Deck Card Detector):
```
item {
  id: 1
  name: 'standing'
}

item {
  id: 2
  name: 'sitting'
}

item {
  id: 3
  name: 'sleeping'
}
```
The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file. For the basketball, shirt, and shoe detector example mentioned in Step 4, the labelmap.pbtxt file will look like:
```
item {
  id: 1
  name: 'basketball'
}

item {
  id: 2
  name: 'shirt'
}

item {
  id: 3
  name: 'shoe'
}
```

### 6.1 Configure training
Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!

Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the faster_rcnn_inception_v2_pets.config file into the \object_detection\training directory. Then, open the file with a text editor. There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

Make the following changes to the faster_rcnn_inception_v2_pets.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).

- Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 3 .
- Line 106. Change fine_tune_checkpoint to:
  - fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

- Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:
  - input_path : "C:/tensorflow1/models/research/object_detection/train.record"
  - label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

- Line 130. Change num_examples to the number of images you have in the \images\test directory.

- Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:
  - input_path : "C:/tensorflow1/models/research/object_detection/test.record"
  - label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

### 6.2 Run the Training
From the \object_detection directory, run the following command to begin training:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
TensorFlow will initialize the training. The initialization can take up to 30 seconds before the actual training begins.
let it train for 3 hours minimum or if the loss is plateau or if the loss is constant at 0.05

### 6.3 If the training if done for 3 Hours 
then press ctrl+c to stop the training.

### 7.0 Export Inference Graph
Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

### 8. Use Your Newly Trained Object Detection Classifier!

The object detection classifier is all ready to go! I’ve written Python scripts to test it out on an image.

Before running the Python scripts, you need to modify the NUM_CLASSES variable in the script to equal the number of classes you want to detect. (For my Human Posture Detector, there are 3 position I want to detect, so NUM_CLASSES = 3.)
so open Object_detection_image.py in an editor and change the NUM_CLASS.

To run any of the scripts, type “idle” in the Anaconda Command Prompt (with the “tensorflow1” virtual environment activated) and press ENTER. This will open IDLE, and from there, you can open any of the scripts and run them.

If everything is working properly, the object detector will initialize for about 10 seconds and then display a window showing any objects it’s detected in the image!

## Built With

* Python
* Tensorflow
* labelImg

## Authors

* **Saket Shetty** - *Initial work* - [saket-shetty](https://github.com/saket-shetty)
* **Varun Shetty** - *Initial work* - [varun-shetty](https://github.com/varunshetty1)
* **Faisal Shaikh**

## Contact
email: shettysaket05@gmail.com


