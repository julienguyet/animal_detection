# Detection of Elephants on Wildlife Pictures

---

## 1. Dataset :elephant:

The data used for this project is the [African Wildlife dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife) from Kaggle by Bianca Ferreira.

It is made of four different classes, but only elephants and rhinos were retained for this exercise. The goal was to recognize elephants and framed them on a given picture, and to not add any annotation if no elephants were present.

<img width="517" alt="image" src="https://github.com/julienguyet/animal_detection/assets/55974674/9f03f73f-dd1f-4b09-bcd5-4d4686b04918">

---

## 2. Modeling :zap:

To perform the task we confronted two models: [YOLOv8](https://yolov8.com/) and [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) from ultranalytics library.

The full code is available in ```notebook/animal_detection_YOLO_RTDETR.ipynb```.
Models were loaded with the following set up:

```python
yolov8_medium = YOLO("yolov8m.pt")

# freeze = how many layers to freeze in context of transfer learning
# patience = how many epochs to wait if NN hits plateau
# batch=-1 means YOLO to maximize batch size by looking at GPU capacity (often 60% so pushed to 80%)
yolov8_medium.train(data=yaml_file_path, epochs=100, imgsz=640, freeze=5, patience=50, batch=0.8) 
```

```python
RTDETR_model = RTDETR('rtdetr-l.pt')

RTDETR_model.train(data=yaml_file_path, epochs=nb_epochs, freeze=5, batch=0.8)
```

:rotating_light: these set up requires a performant NVIDIA GPU :rotating_light: 
I highly suggest reducing the batch allocation or leave it to none (auto allocation) if your computer has limited capacity

---

## 3. Results :microscope:

We obtained a Recall of ~0.90 on both train and validation sets for YOLOv8 with a training of 100 epochs in 15 minutes. TR-DETR showed more difficulties to reach good results but score a 0.70 recall in only 5 epochs (4 minutes of training) which demonstrates strong potential. 

Below are some successful examples:

<img width="639" alt="image" src="https://github.com/julienguyet/animal_detection/assets/55974674/ef33f068-7052-4f9b-bd1f-229a94455060">

And some errors:

<img width="942" alt="image" src="https://github.com/julienguyet/animal_detection/assets/55974674/7ec5a9df-cc25-4c81-9003-107611c297ef">


Confusion Matrix and more detailled analysis are available in the notebook. 
