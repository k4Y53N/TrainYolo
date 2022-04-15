# TrainYolo
### 訓練自定義模型
##### 修改sys.ini
```ini
[Annotations]
train_set_dir = data/train2014
train_annotation_path = data/instances_train2014.json
test_set_dir = data/test2014
test_annotation_path = data/instances_test2014.json

[Save_dir]
checkpoints = checkpoints
weights = weights
configs = configs
logs = logs
train_processed_data = data/bbox/train
test_processed_data = data/bbox/test
```
sys.ini內包含了COCO資料集路徑、COCO資料集標記文件路徑、存檔位置

##### 建立一個txt檔，每一行紀錄要辨識的物件名稱
```
person
car
horse
```
##### 產生YOLO設定檔(.json)
```commandline
python3 makeYoloConfig.py -n myModelName -c classes.txt -s 416 -sc 0.5 -bs 4 -ep 50 -ts 2000 -vs 100
```
設定檔內包含模型資訊、物件種類、尺寸、存檔位置、訓練資訊等參數
```json
{
    "name": "myModelName",
    "model_path": "checkpoints/myModelName",
    "weight_path": "weights/myModelName.h5",
    "logdir": "logs/myModelName",
    "frame_work": "tf",
    "model_type": "yolov4",
    "size": 416,
    "tiny": false,
    "max_output_size_per_class": 40,
    "max_total_size": 50,
    "iou_threshold": 0.5,
    "score_threshold": 0.5,
    "YOLO": {
        "CLASSES": [
            "person",
            "car",
            "horse"
        ],
        "ANCHORS": [
            5,
            13,
            21,
            34,
            38,
            108,
            50,
            158,
            113,
            121,
            66,
            219,
            351,
            95,
            184,
            384,
            339,
            373
        ],
        "ANCHORS_V3": [
            5,
            13,
            21,
            34,
            38,
            108,
            50,
            158,
            113,
            121,
            66,
            219,
            351,
            95,
            184,
            384,
            339,
            373
        ],
        "ANCHORS_TINY": [
            12,
            23,
            45,
            108,
            69,
            175,
            351,
            95,
            184,
            384,
            339,
            373
        ],
        "STRIDES": [
            8,
            16,
            32
        ],
        "STRIDES_TINY": [
            16,
            32
        ],
        "XYSCALE": [
            1.2,
            1.1,
            1.05
        ],
        "XYSCALE_TINY": [
            1.05,
            1.05
        ],
        "ANCHOR_PER_SCALE": 3,
        "IOU_LOSS_THRESH": 0.5
    },
    "TRAIN": {
        "ANNOT_PATH": "data/bbox/train/myModelName.bbox",
        "BATCH_SIZE": 4,
        "INPUT_SIZE": 416,
        "DATA_AUG": true,
        "LR_INIT": 0.001,
        "LR_END": 1e-06,
        "WARMUP_EPOCHS": 2,
        "INIT_EPOCH": 0,
        "FIRST_STAGE_EPOCHS": 0,
        "SECOND_STAGE_EPOCHS": 50,
        "PRETRAIN": null
    },
    "TEST": {
        "ANNOT_PATH": "data/bbox/test/myModelName.bbox",
        "BATCH_SIZE": 4,
        "INPUT_SIZE": 416,
        "DATA_AUG": false,
        "SCORE_THRESHOLD": 0.5,
        "IOU_THRESHOLD": 0.5
    }
}
```
##### 開始訓練
```commandline
python3 train.py configs/myModelName.json
```
如果在訓練過程中意外退出，直接再次執行這行指令即可，訓練結束後下載權重(.h5)以及設定檔(.json)到Jetson nano相對應的資料夾位置
