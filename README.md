# detectCheckerboard
Utilize Tensorflow to detect checkerboards

Assuming that you have your data saved in 'annotations' and 'images' folders. You can check 'examples' folder to see how they look like
Inside project folder, do the followings steps

1. Install TensorFlow API

```
    cd <project_folder> 
    git clone https://github.com/tensorflow/models.git
    cd models/research/
    protoc object_detection/protos/*.proto --python_out=.
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

2. Split the data into training and evaluation sets.
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

3. Convert data to [TFRecord](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py) with some 
necessary modifications. From project folder, run
```
    python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
    python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
```
  The process will take a while. An error "No module named "object_detection"" arises, you can either move the script and 
  run it inside the module with suitable modification for the argument of *--csv_input* or copy the folder "object_detection"
  from "models/research/" to the running directory.
 
4. Create label map [*"label_map.pbtxt"*](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)
```
    item {
        id = 1
        name = "checkerboard"
    }
```

5. Download a model from [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), 
e.g, ssd_mobilenet_v1_coco.

5. Choose a model and configure the pipeline accordingly
```
    tf_record_input_reader {
        input_path: "path/to/the/project/data/train.record"
    }
    label_map_path: "path/to/the/project/data/label_map.pbtxt"
```
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
    fine_tune_checkpoint: "path/to/the/project/.../tmp/model.ckpt-#####"
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
9. The result can be illustrated using another batch of testing images and run the script **object_detection.py**. 
In the script, modify the following code in accordance with your project: 
```
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    ...
    PATH_TO_CKPT = 'path/to/the/project' + '/training/train/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('path/to/the/project', 'data/label_map.pbtxt')
    ...
    PATH_TO_TEST_IMAGES_DIR = 'path/to/the/project' + 'data/test/checkerboard_test'
```