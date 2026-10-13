from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from traffic_simulation import *
from yolo_traffic import *
import cv2
import numpy as np
import tensorflow as tf

main = tkinter.Tk()
main.title("Edge-AI Enabled Video Analytics in Smart Cities")
main.geometry("1300x1200")

global filename

net = cv2.dnn.readNetFromCaffe("yolo-coco/MobileNetSSD_deploy.prototxt.txt","yolo-coco/MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def yoloTrafficDetection():
    global filename
    filename = filedialog.askopenfilename(initialdir="Videos")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    runYolo(filename)
   
def runSimulation():
    sim = Simulation()
    sim.runSimulation()

def ssdDetection(image_np):
    count = 0
    (h, w) = image_np.shape[:2]
    ssd = tf.Graph()
    with ssd.as_default():
        od_graphDef = tf.GraphDef()
        with tf.gfile.GFile('yolo-coco/frozen_inference_graph.pb', 'rb') as file:
            serializedGraph = file.read()
            od_graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(od_graphDef, name='')
    with ssd.as_default():
        with tf.Session(graph=ssd) as sess:
            blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)),0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    if (confidence * 100) > 0:
                        if CLASSES[idx] == "bicycle" or CLASSES[idx] == "bus" or CLASSES[idx] == "car":
                            count = count + 1
                            label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                            cv2.rectangle(image_np, (startX, startY), (endX, endY),COLORS[idx], 2)
                            cv2.putText(image_np, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(image_np, "Detected Count : "+str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    return image_np                    

def extensionSingleShot():
    global filename
    filename = filedialog.askopenfilename(initialdir="Videos")
    pathlabel.config(text=filename)
    video = cv2.VideoCapture(filename)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result = ""
    while(True):
        ret, frame = video.read()
        frame = ssdDetection(frame)
        cv2.imshow("Frame", frame)
        print(ret)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()    

    
font = ('times', 16, 'bold')
title = Label(main, text='Edge-AI Enabled Video Analytics in Smart Cities')
title.config(bg='light cyan', fg='pale violet red')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
simulationButton = Button(main, text="Run Traffic Simulation", command=runSimulation)
simulationButton.place(x=50,y=100)
simulationButton.config(font=font1)  

yoloButton = Button(main, text="Run Extension Yolo Traffic Detection & Counting", command=yoloTrafficDetection)
yoloButton.place(x=460,y=100)
yoloButton.config(font=font1) 

pathlabel = Label(main)
pathlabel.config(bg='light cyan', fg='pale violet red')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='snow3')
main.mainloop()
