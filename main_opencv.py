import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.25
CONFIDENCE_THRESHOLD = 0.25

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

BLACK  = (0,0,0)
BLUE   = (255,0,0)

# Elimina le zone non di interesse dall' immagine applicando una maschera
def define_Region_of_Interest(image):              
    mask = np.zeros(image.shape[:2], np.uint8)
    mask = cv2.bitwise_not(mask)
    return image

# Disegna i risultati di ogni oggetto riconosciuto 
def draw_label(im, label, x, y):
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, (0,255,255), THICKNESS, cv2.LINE_AA)

# Tramite una lista di coordinate viene creata una mappa di densità
def DensityMap(x_array,y_array,count):
    xmin = x_array.min() - 50
    xmax = x_array.max() + 50
    ymin = y_array.min() - 50
    ymax = y_array.max() + 50

    X, Y = np.mgrid[xmin:xmax:400j, ymin:ymax:400j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x_array, y_array])
    # Applicazione di kernel Gaussiani per la stima di densità
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    ax.plot(x_array, y_array, 'k.', markersize=3)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.pcolormesh(X, Y, Z.reshape(X.shape), shading='auto' ) #, cmap='jet')
    if mode == "v":
        fig.savefig(f"./results/video/densityMaps/densitymap_{count}.jpg")
    else :
        fig.savefig(f"./results/image/densitymap.jpg")
    plt.close(fig)
    
# Pre-processamento dei risultati
def pre_process(input_image, net): 
    # Crea e imposta l'input della rete
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    # Risultati della rete
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs

# Elaborazione risultati
def post_process(img,detections):
    
    confidences = []
    boxes = []
    
    rows = detections[0].shape[1]
    image_height, image_width = img.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    for r in range(rows):
        # Risultati di un oggetto
        row = detections[0][0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5]
            if (classes_scores > SCORE_THRESHOLD):
                confidences.append(confidence)
                cx, cy, w, h = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                width = int(row[2] * x_factor)
                height = int(row[3] * y_factor)
                x = int((cx - w / 2) * x_factor)
                y = int((cy - h / 2) * y_factor)
                box = np.array([x, y, width, height])
                boxes.append(box)
    
    return  confidences,boxes

# Funzione che ottenuta un'immagine costriusce la mappa di densità tramite le funzioni viste
def detect(img,masked,count):
    image_height, image_width = img.shape[:2]
    detections = pre_process(masked, net)
    confidences, boxes = post_process(img,detections)
    center_points = []
    #Non-maximum-suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    print(f"Numero di veicoli individuati : {len(indices)}")
    print()
    # Vengono estratte le coordinate delle bbox e disegnate
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        center_points.append((x,abs(y-image_height)))
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(img, (x, y), (x + w, y + h), BLUE, 2)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
    csv_data.append([count-1,len(center_points)])
    # Creazione di due array contenenti rispettivamente le coordinate x e y separate 
    xs = np.array([xi for xi, yi in center_points])
    ys = np.array([yi for xi, yi in center_points])
    if len(xs) > 2:
        DensityMap(xs,ys,count)

    
# Ramo per elaborare video
def video(path):

    #######################################################
    # ELIMINA I RISULTATI PRECEDENTI
    video_frame_folder = "./results/video/frameSaved"

    for filename in os.listdir(video_frame_folder):
        file_path = os.path.join(video_frame_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    video_dm_folder = "./results/video/densityMaps"

    for filename in os.listdir(video_dm_folder):
        file_path = os.path.join(video_dm_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    ########################################################

    print("\n+++++++++++++++++++++++++++++++++++++++++++++++++")

    cap = cv2.VideoCapture(path)

    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    framecount = 0
    count = 0

    while True : 

        ret, frame = cap.read()
        if ret is not True:
            break
        framecount += 1

        if framecount == (framerate * 5) :  
            print("WAITING !")

            framecount = 0
            count += 1

            frame = cv2.resize(frame, (1280, 780))

            masked = define_Region_of_Interest(frame)

            detected = detect(frame,masked,count)

            cv2.imwrite(f"./results/video/framesaved/frame_{count}.jpg", frame)

            if cv2.waitKey(1) == ord('q'):
                break
    
    print("Results saved in './results/video/' folder")
    print("\nENDING")
    print("++++++++++++++++++++++++++++++++++++++++++\n")

    # Scrive i risultati sul file csv
    with open("./results/video/data.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
    
    # Dai dati raccolti viene generato un grafico che descrive il numero di veicoli individuati nel tempo
    with open("./results/video/data.csv", "r") as file:
        x = np.array([])
        y = np.array([])
        reader = csv.reader(file, delimiter = ',')
        next(reader)
        for row in reader:
            x = np.append(x,int(row[0]))
            y = np.append(y,int(row[1]))
        media = sum(y) / len(x)
        plot1 = plt.plot(x,y,'blue',label="Numero Veicoli")
        plot2 = plt.plot(x,[media for i in range(len(x))],'red',label="Media")
        plt.xlabel('Istanti Temporali')           
        plt.ylabel('Numero Veicoli')
        plt.text(len(x)/2,0, f"Media = {int(media)}",fontsize=11)
        plt.legend()
        plt.grid(True)
        plt.axis([0,x.max(),0,y.max()+10])
        plt.savefig("./results/video/n-vehicle-trend.jpg")
    

    cv2.namedWindow('window')
    cv2.destroyAllWindows()

# Ramo per elaborare immagini
def image(path):

    #######################################################
    # ELIMINA I RISULTATI PRECEDENTI
    image_folder = "./results/image"

    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    #######################################################
    
    img = cv2.imread(path)

    masked = define_Region_of_Interest(img)

    detect(img,masked,0)

    cv2.imwrite("./results/image/img.jpg", img)

    print("Results saved in './results/images/' folder\n")


####### MAIN ########

# File csv che raccoglie i dati dalle immagini elaborate:
# usato solo nella computazione di un video
csv_data = [["Interval","N_Vehicles"]]

# Apro il file contenente i nomi delle classi, nel mio caso vi è solo un nome
with open("custom.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Carico il modello della rete
net = cv2.dnn.readNetFromONNX("weights.onnx")

# Si sceglie la modalità di esecuzione dello script
mode = input("\nVideo(v) or Image(i) ? ")

try:
    if mode == "v":
        video("./test/...")     # Percorso del video
    elif mode == "i":
        image("./test/...")     # Percorso dell'immagine
except FileNotFoundError:
    print("File non trovato")
