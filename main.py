from ultralytics import YOLO 
import random
import cv2

# #NOTE: prefered training set
# from datasets import load_dataset
# #Loading dataset
# load_dataset("visual_genome", "objects_v1.2.0")

    #loading model
model  = YOLO('yolov8n.pt')

def detection(frame):
    #prediction
    results = model.predict(source=frame)[0]
    #COLORING NAMES OF OBJECTS DETECTED
    names = list(set(results.boxes.cls.tolist()))
    colors = [tuple([random.randint(0,255) for _ in range(3)]) for _ in range(len(names))]
    cls_id_col = dict(zip(names,colors))
    #print(cls_id_col)
    
    #Getting Info of objects detected
    for result in results:
        box = result.boxes 
        cords = box.xyxy[0].tolist()
        cls = box.cls[0].item()
        cords = [round(x) for x in cords]
        name = result.names[box.cls[0].item()]
        conf = round(box.conf[0].item(),2)
        
        #thresholding on the basis of confidence:
        if(conf>=0.4):
        #for rectangle on object
            cv2.rectangle(frame,(int(cords[0]),int(cords[1])),(int(cords[2]),int(cords[3])),color=cls_id_col[cls],thickness=2)
        #text background color:
            text_size,baseline = cv2.getTextSize(name,cv2.FONT_HERSHEY_COMPLEX,fontScale=0.4,thickness=1)
            cv2.rectangle(frame, (int(cords[0]), int(cords[1] - text_size[1] - baseline)), (int(cords[0] + text_size[0]), int(cords[1])), (50,50,50), -1)
        
        #to add text
            cv2.putText(frame,name,(int(cords[0]),int(cords[1])),cv2.FONT_HERSHEY_COMPLEX,fontScale=0.4,color=(255,255,255),thickness=1)
            
    return frame

#the image preprocessor
def image_processor():
    img = cv2.imread('dataset/room.jpg')
    img = cv2.resize(img,(0,0), fx =0.2,fy =0.2)
    img = detection(img)
    cv2.imshow("Detected",img)
    cv2.waitKey(0)

#the video prepropcessor
def video_processor():
    vid = cv2.VideoCapture('dataset/people.mp4')

def live_video_processor():
    vid = cv2.VideoCapture(0)
    while(vid.isOpened()):
        ret,frame = vid.read()
        
        if ret is not False:
            img = detection(frame)
            cv2.imshow('Video',img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    vid.release()

live_video_processor() 
#video_processor()
#image_processor()
    