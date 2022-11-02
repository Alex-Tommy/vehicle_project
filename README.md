# Vehicle Density Map for Traffic Managment
Progetto che crea mappe 2D di densità che rappresentano la predisposizione dei veicoli in una determinata immagine.
- SISTEMA DI OBJECT DETECTION : YOLOv5m
- DATASET : UAVDT Dataset
- ALLENAMENTO
- RISULTATI
---
###  YOLOv5m
Per maggiori informazioni sul modello usato visualizzare questo [link](https://github.com/ultralytics/yolov5)

---
###  UAVDT-M Dataset 
Il dataset è composto da circa 40000 immagini che hanno le seguenti caratteristiche.

![](https://github.com/Alex-Tommy/vehicle_project/blob/main/repo-images/dataset-composition.png)

Per maggiori informazioni sul dataset usato aprire questo [link](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5)

Nella cartella `scripts` sono presenti i file che ho utilizzato per formattare il dataset nel formato YOLO:
* `organise_image_folders.py` copia le immagini in un unica cartella
* `organise_txt_labels.py` formatta le annotazioni nel formato YOLO `class-object  x-center  y-center  width  height` compresi in [0,1]
* `split_train_val.py` crea un training set e un validation set

---
###  ALLENAMENTO
Allenamento svolto sulla piattaforma Google Colab seguendo il notebook `training.ipynb`

---
###  RISULTATI


