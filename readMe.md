## Request computational resource
`
srun --mail-type=ALL --mail-user=xxx@ufl.edu --nodes=1 --ntasks=1 --partition=gpu --gpus=a100:2 --mem=40G --time=02:00:00 --pty bash -i
`

## Delete previous labelling data
`
rm -r data.yaml README.dataset.txt README.roboflow.txt train/ valid/
`

## Start training
`
yolo task=detect mode=train model=yolov8x.pt data=data.yaml epochs=100 imgsz=640
`

`
yolo task=detect mode=segment model=yolov8x-seg.pt data=data.yaml epochs=100 imgsz=640
`


## Inference argument

### https://docs.ultralytics.com/modes/predict/#inference-arguments
`
yolo predict model=AI-8-9-10-11-X.pt source="IMG_8428.JPG" save=True conf=0.90 boxes=False
`