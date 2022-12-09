import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import stats
import seaborn as sns
import os
import csv

def define_Region_of_Interest(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    mask = cv2.bitwise_not(mask)
    cv2.rectangle(mask, (1000,0), (image.shape[1],400), 0, -1)
    return cv2.bitwise_and(image, image, mask=mask)

def DensityMap(x_array, y_array,count,mode):
    xmin = x_array.min() - 50
    xmax = x_array.max() + 50
    ymin = y_array.min() - 50
    ymax = y_array.max() + 50

    X, Y = np.mgrid[xmin:xmax:400j, ymin:ymax:400j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x_array, y_array])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
    ax.plot(x_array, y_array, 'k.', markersize=3)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.pcolormesh(X, Y, Z.reshape(X.shape), shading='auto' , cmap='jet')
    if mode == "v":
        fig.savefig(f"./results/video/densityMaps/densitymap_{count}.jpg")
    else :
        fig.savefig("./results/images/densitymap.jpg")
    plt.close(fig)
    

def detect_points(image,masked,count):
    img_h, img_w = masked.shape[:2]
    results = model(masked)
    df = results.pandas().xyxy[0]
    print(f"Veicoli individuati : {df.shape[0]}")
    xs = np.array([])
    ys = np.array([])
    data = np.array([])
    for index, row in df.iterrows():
        xCenter = (row['xmin'] + row['xmax']) / 2
        yCenter = (row['ymin'] + row['ymax']) / 2
        xs = np.append(xs,int(xCenter))
        ys = np.append(ys,int(abs(yCenter - img_h)))
        cv2.rectangle(image,(int(row['xmin']),int(row['ymin'])),(int(row['xmax']),int(row['ymax'])),(255,0,0),2)
    csv_data.append([count-1,df.shape[0]])
    if df.shape[0] > 2 :
        DensityMap(xs,ys,count,mode)


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

            detect_points(frame,masked,count)

            cv2.imwrite(f"./results/video/framesaved/frame_{count}.jpg", frame)

            if cv2.waitKey(1) == ord('q'):
                break
    
    print("Results saved in './results/video/' folder")
    print("\nENDING")
    print("++++++++++++++++++++++++++++++++++++++++++")
    print()

    with open("./results/video/data.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

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
        plt.text(len(x)/2,0, f"Media = {int(media)}",fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.axis([0,x.max(),0,y.max()+10])
        plt.savefig("./results/video/n-vehicle-trend.jpg")
    

    cv2.namedWindow('window')
    cv2.destroyAllWindows()


def image(path):

    #######################################################
    # ELIMINA I RISULTATI PRECEDENTI
    image_folder = "./results/images"

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

    detect_points(img,masked,0)

    cv2.imwrite("./results/images/img.jpg", img)

    print("Results saved in './results/images/' folder")
    print()


####### MAIN ########
# yolov5-master è la directory clonata di yolov5 --> https://github.com/ultralytics/yolov5
model = torch.hub.load("../yolov5-master", 'custom', path='300.pt', source='local') 

csv_data = [["Interval","N-vehicles"]]

print()
mode = input("Video(v) or Image(i) ? ")

try:
    if mode == "V" or mode == "v":
        video("./test_video/2.mp4") 
    elif mode == "I" or mode == "i":
        image("./test_images/13.jpg")
except FileNotFoundError :
    print("File non trovato")