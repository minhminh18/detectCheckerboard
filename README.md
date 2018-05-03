# detectCheckerboard
Utilize Tensorflow to detect checkerboards

Assuming that you have your data saved in 'annotations' and 'images' folders. You can check 'examples' folder to see how they look like
Inside project folder, do the followings steps

1. Install TensorFlow API
git clone https://github.com/tensorflow/models.git
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

2. Split the data into training and evauation using split_labels.py
3. Convert data to TFRecord
https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py with necessary modifications
From project folder, run
python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record

4. Download a model from model zoo
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

5. Choose a model and configure the pipeline accordingly
```
tf_record_input_reader {
  input_path: "/usr/home/username/data/train.record"
}
label_map_path: "/usr/home/username/data/label_map.pbtxt"
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
fine_tune_checkpoint: "/usr/home/username/tmp/model.ckpt-#####"
from_detection_checkpoint: true
gradient_clipping_by_norm: 10.0
data_augmentation_options {
  random_horizontal_flip {
  }
}
```

6. Run training, data folder contains the train.record file
python models/research/object_detection/train.py --logtostderr --train_dir=${PATH_TO_TRAIN_DIR} \
	--pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG}
In this project, the pipeline_config can be found under 'data' folder

7. Stop the training with Ctrl+C when the loss reaches about 1

8. Export the graph for reference
python models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${PATH_TO_TRAIN_DIR}/model.ckpt-3179 \
    --output_directory object_detection_graph
