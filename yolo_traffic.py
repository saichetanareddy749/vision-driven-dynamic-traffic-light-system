# import the necessary packages
import numpy as np
import imutils
import time
from scipy import spatial
import cv2
import os

# All these classes will be counted as 'vehicles'
list_of_vehicles = ["bicycle","car","motorbike","bus","truck","train"]

FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416

labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

preDefinedConfidence = 0.35
preDefinedThreshold = 0.25

np.random.seed(42)
COLORS = np.random.randint(0,255,size=(len(LABELS),3),dtype="uint8")

# ------------------------------------------------
# DISPLAY VEHICLE COUNT
# ------------------------------------------------
def displayVehicleCount(frame, vehicle_count):
    cv2.putText(frame,
        "Detected Vehicles: "+str(vehicle_count),
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2)

# ------------------------------------------------
# TRAFFIC SIGNAL CONTROL (YOUR LOGIC - SAME)
# ------------------------------------------------
def trafficSignalControl(frame, vehicle_count):

    if vehicle_count > 50:
        traffic="HIGH"
        signal="GREEN"
        green_time=40
        color=(0,255,0)

    elif vehicle_count > 30:
        traffic="HIGH"
        signal="GREEN"
        green_time=30
        color=(0,255,0)

    elif vehicle_count > 15:
        traffic="MEDIUM"
        signal="YELLOW"
        green_time=20
        color=(0,255,255)

    else:
        traffic="LOW"
        signal="RED"
        green_time=10
        color=(0,0,255)

    cv2.putText(frame,"Traffic Density: "+traffic,(20,70),
        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(frame,"Signal: "+signal,(20,100),
        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.putText(frame,"Recommended Green Time: "+str(green_time)+" sec",
        (20,130),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

# ------------------------------------------------
# DRAW DETECTION BOXES
# ------------------------------------------------
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x,y)=(boxes[i][0],boxes[i][1])
            (w,h)=(boxes[i][2],boxes[i][3])

            color=[int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            text="{}: {:.2f}".format(LABELS[classIDs[i]],confidences[i])
            cv2.putText(frame,text,(x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

# ------------------------------------------------
# TRACK VEHICLES
# ------------------------------------------------
def boxInPreviousFrames(previous_frame_detections,current_box,current_detections):

    centerX,centerY,width,height=current_box
    dist=np.inf

    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list=list(previous_frame_detections[i].keys())

        if len(coordinate_list)==0:
            continue

        temp_dist,index=spatial.KDTree(coordinate_list).query([(centerX,centerY)])

        if temp_dist<dist:
            dist=temp_dist
            frame_num=i
            coord=coordinate_list[index[0]]

    if dist>(max(width,height)/2):
        return False

    current_detections[(centerX,centerY)]=previous_frame_detections[frame_num][coord]
    return True

# ------------------------------------------------
# COUNT VEHICLES
# ------------------------------------------------
def count_vehicles(idxs,boxes,classIDs,vehicle_count,previous_frame_detections,frame):

    current_detections={}

    if len(idxs)>0:
        for i in idxs.flatten():

            (x,y)=(boxes[i][0],boxes[i][1])
            (w,h)=(boxes[i][2],boxes[i][3])

            centerX=x+(w//2)
            centerY=y+(h//2)

            if LABELS[classIDs[i]] in list_of_vehicles:

                current_detections[(centerX,centerY)]=vehicle_count

                if not boxInPreviousFrames(previous_frame_detections,
                                           (centerX,centerY,w,h),
                                           current_detections):
                    vehicle_count+=1

                cv2.putText(frame,str(vehicle_count),(centerX,centerY),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

    return vehicle_count,current_detections

# ------------------------------------------------
# RUN YOLO DETECTION
# ------------------------------------------------
def runYolo(inputVideoPath):

    print("[INFO] loading YOLO...")

    net=cv2.dnn.readNetFromDarknet(configPath,weightsPath)
    ln=net.getLayerNames()
    ln=[ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

    videoStream=cv2.VideoCapture(inputVideoPath)

    video_width=int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height=int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # OUTPUT SETUP
    if not os.path.exists("output"):
        os.makedirs("output")

    output_path = "output/traffic_output.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (900,600))

    previous_frame_detections=[{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
    vehicle_count=0

    while True:

        grabbed,frame=videoStream.read()
        if not grabbed:
            break

        blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
        net.setInput(blob)
        layerOutputs=net.forward(ln)

        boxes=[]
        confidences=[]
        classIDs=[]

        for output in layerOutputs:
            for detection in output:

                scores=detection[5:]
                classID=np.argmax(scores)
                confidence=scores[classID]

                if confidence>preDefinedConfidence:

                    box=detection[0:4]*np.array([video_width,video_height,video_width,video_height])
                    (centerX,centerY,width,height)=box.astype("int")

                    x=int(centerX-(width/2))
                    y=int(centerY-(height/2))

                    boxes.append([x,y,int(width),int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs=cv2.dnn.NMSBoxes(boxes,confidences,preDefinedConfidence,preDefinedThreshold)

        drawDetectionBoxes(idxs,boxes,classIDs,confidences,frame)

        vehicle_count,current_detections=count_vehicles(
            idxs,boxes,classIDs,vehicle_count,previous_frame_detections,frame)

        displayVehicleCount(frame,vehicle_count)
        trafficSignalControl(frame,vehicle_count)

        frame=cv2.resize(frame,(900,600))

        # SAVE FRAME
        out.write(frame)

        cv2.imshow("Traffic Analysis",frame)

        # 🚨 STOP CONDITION
        if vehicle_count > 80:
            print("[INFO] Vehicle count exceeded 80. Saving video and stopping...")
            break

        if cv2.waitKey(1)&0xFF==ord("q"):
            break

        previous_frame_detections.pop(0)
        previous_frame_detections.append(current_detections)

    videoStream.release()
    out.release()
    cv2.destroyAllWindows()

    print("[INFO] Video saved at:", output_path)