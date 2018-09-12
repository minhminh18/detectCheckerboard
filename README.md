# detectCheckerboard
Utilize Tensorflow to detect checkerboards in the images with fine tuned model.

## Prerequisite
Your system have **tensorflow** installed. It is recommended to install with *anaconda*. Additionally, using CUDA will speed up 
the training process.

## Installation
Assuming that you have your data saved in *'annotations'* and *'images'* folders. You can check *'examples'* folder to see how they look like
inside project folder, do the followings steps (remember to replace *<path_to_the_project>* accordingly):

1. Install TensorFlow API

```
    cd <project_folder> 
    git clone https://github.com/tensorflow/models.git
    cd models/research/
    protoc object_detection/protos/*.proto --python_out=.
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

## Training
1. Split the data into training and evaluation sets.
* Create a folder name *'data'* inside the working directory.
* Use the script __xml_to_csv.py__ to read all names of the images.
```
    python xml_to_csv.py  
```
The console will output *"Successfully converted xml to csv."*

* Run the script __split_labels.py__
```
    python split_labels.py
``` 
The console will show *"Successfully splitted the labels."*
The data folder will look like
```
    data
    |-- checkerboard_labels.csv
    |-- test_labels.csv
    |-- train_labels.csv
```

2. Convert data to [TFRecord](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py) with some 
necessary modifications. [Converting data](https://www.oreilly.com/ideas/object-detection-with-tensorflow)
 provides the training process a high pace of accessing data comparing to reading
 directly each image.From project folder, run
```
    python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
    python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
```
  The process will take a while. An error "No module named "object_detection"" arises, you can either move the script and 
  run it inside the module with suitable modification for the argument of *--csv_input* or copy the folder "object_detection"
  from "models/research/" to the running directory.
 
3. Create label map [*"label_map.pbtxt"*](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)
```
    item {
        id = 1
        name = "checkerboard"
    }
```

4. Download a model from [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), 
e.g, ssd_mobilenet_v1_coco. They are stored as [checkpoints](https://www.tensorflow.org/guide/checkpoints) 
which are different versions of the model created during training.


5. Choose a model and configure the pipeline accordingly
```
    model{
      ssd{
         num_classes: 1
    ...
       }
    }
    train_config:{
      fine_tune_checkpoint: "<path_to_the_checkpoint>/model.ckpt"
    from_detection_checkpoint: true
    ... 
    }
    train_input_reader:{
        tf_record_input_reader {
            input_path: "<path_to_the_project>/data/train.record"
        }
        label_map_path: "<path_to_the_project>/data/label_map.pbtxt"
    }
    eval_input_reader: {
        tf_record_input_reader {
            input_path: "<path_to_the_project>/data/test.record"
        }
        label_map_path: "<path_to_the_project>/data/label_map.pbtxt"
    }
```
* The number of class needs to be changed in accordance with how many object we intend to train. In this project, it was set to 1.
* The directories are modified in accordance with the location of the files.
* The checkpoint typically includes three files: 
  * checkpoint
  * model.ckpt-{checkpoint_number}.data-0000-of-00001, 
  * model.ckpt-{checkpoint_number}.meta, and
  * model.ckpt-{checkpoint_number}.ckpt.index.
They record the information of the training process including weights. If a training process in postponed unintentionaly, it can be continue from the previous saved checkpoint.

Configuring the trainer
```
    batch_size: 1
    optimizer {
      momentum_optimizer: {
        learning_rate: {
          manual_step_learning_rate {
            initial_learning_rate: 0.0002
            schedule {
              step: 0
              learning_rate: .0002
            }
            schedule {
              step: 900000
              learning_rate: .00002
            }
            schedule {
              step: 1200000
              learning_rate: .000002
            }
          }
        }
        momentum_optimizer_value: 0.9
      }
      use_moving_average: false
    }
    fine_tune_checkpoint: "<path_to_the_project>/.../tmp/model.ckpt-#####"
    from_detection_checkpoint: true
    gradient_clipping_by_norm: 10.0
    data_augmentation_options {
      random_horizontal_flip {
      }
    }
```

6. Run training, data folder contains the *train.record* file
```bash
    python models/research/object_detection/train.py --logtostderr --train_dir=${PATH_TO_TRAIN_DIR} \
        --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG}
```
In this project, the *pipeline_config* can be found under *'data'* folder

7. Stop the training with Ctrl+C when the loss reaches approximately 1.

8. Export the graph for reference
```bash
    python models/research/object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG} \
        --trained_checkpoint_prefix ${PATH_TO_TRAIN_DIR}/model.ckpt-3179 \
        --output_directory object_detection_graph
```
The function generated a graph in pb ([protocol buffer](https://github.com/protocolbuffers/protobuf)) 
format which serialize structured data, define parameters for the callable methods. The [graph](https://github.com/protocolbuffers/protobuf) 
contains the model architecture and weights. Multiple checkpoints can be tested for the best performance.
9. The result can be illustrated using another batch of testing images and run the script **object_detection.py**. 
In the script, modify the following code in accordance with your project: 
```
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    ...
    PATH_TO_CKPT = <path_to_the_project> + '/training/train/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join(<path_to_the_project>, 'data/label_map.pbtxt')
    ...
    PATH_TO_TEST_IMAGES_DIR = <path_to_the_project> + 'data/test/checkerboard_test'
```

## Result
* The training stage consumes in general about more than one hour. 
* In training TensorFlow model converges quickly. It reduced from approximately 12 to 3 after about 200 steps. After that, 
the model required more time to reduce loss. After about 4000 steps, the convergence hardly decreased. The loss of almost 1.0 was achieved after more than 10000 steps. However, sometimes it could increase which may be a result of overfitting.


## Problems and troubleshooting
1. Sufficiency of images:

A typical training process requires upto thousand of images and the [variation of the background](http://aiweirdness.com/post/171451900302/do-neural-nets-dream-of-electric-sheep "electric sheep") 
influences significantly
on the result of the training. Hence, it is recommended to prepare enough images. A method to test your batch of images is 
using classification model. After [preparing the images](https://github.com/minhminhng/preparing_training_images "preparing images"), you can 
run  
```bash
    python tensorflow/examples/image_retraining/retrain.py\
    --image_dir ~/<path_to_the_project>/data/ --learning_rate=0.0001\ --testing_percentage=20 --validation_percentage=20 
    --train_batch_size=32 --validation_batch_size=-1  --flip_left_right True --random_scale=30\
    --random_brightness=30 --eval_step_interval=100\ --how_many_training_steps=1000  --architecture mobilenet_1.0_224

```
The parameters can be modified corresponding to the data set:
* learning rate: commonly sufficient at 0.0001. A lower learning rate can output better result but consume more time.
* testing_percentage: normally chosen at 20 to 25 percents.
* train_batch_size: smaller value takes longer time but larger value can hang the process if the processor can not 
handle a large amount of data.
* flip_left_right: set to True to increase the variation of the images. However, checkerboard is a symmetric pattern, 
this parameter is not necessary.
* how_many_training_steps: the higher this value, the higher the accuracy of the classification result but it will 
saturate at some level meaning that the accuracy of the model can not be improved higher with the current data set
* architecture: different architectures of MobileNet can be tried. The first value is the width multiplier which 
can be 1.0, 0.75, 0.50, or 0.25. The end value is the resolution of the image which can be 224, 192, 160, or 128. 
The higher values of width multiplier and resolution can result in better classification accuracy but the training may 
consume more time as a trade-off. Choosing those values is typically dependent on the size of the data set, the requirement 
of accuracy, the experience of the user.

More information can be found at <https://hackernoon.com/creating-insanely-fast-image-classifiers-with-mobilenet-in-tensorflow-f030ce0a2991?gi=71ea783ae893>
* Training process took about 20 minutes on a computer with GPU support.
* The accuracy of the training and testing processes were achieved as about 70 %.
* A typical training process requires about 3 000 to 5 000 images. Normally, the larger the database, the better the accuracy.


2. *ImportError: No module named ‘object_detection’*
* Run the script from inside the module, or
* Copy the folder *‘object_detection’* from *models/research/’* to the directory from which the script is run

3. *ImportError: No module named ‘nets’*
=> From directory *models/research/* do
```
    export PYTHON_HOME=$PYTHON_HOME:`pwd`:`pwd`/slim
```

4. *TypeError: pred must be a Tensor, or a Python bool, or 1 or 0. Found instead: None*

change 109 line in **ssd_mobilenet_v1_feature_extractor.py**:
from
```
       is_training=None, regularize_depthwise=True)):
```
to
```
      is_training=True, regularize_depthwise=True)):
```
