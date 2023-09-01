from ultralytics import YOLO 
import random

# #NOTE: prefered training set
# from datasets import load_dataset
# #Loading dataset
# load_dataset("visual_genome", "objects_v1.2.0")

#loading model
model  = YOLO('yolov8n.pt')

results = model.predict(source='dataset/room.jpg')[0]

#COLORING NAMES OF OBJECTS DETECTED
names = list(set(results.boxes.cls.tolist()))
colors = [tuple([random.randint(0,255) for _ in range(3)]) for _ in range(len(names))]
cls_id_col = dict(zip(names,colors))
print(cls_id_col)
#Getting Info of objects detected
# for result in results:
#     box = result.boxes 
#     cords = box.xywh[0].tolist()
#     cords = [round(x) for x in cords]
#     name = result.names[box.cls[0].item()]
#     conf = round(box.conf[0].item(),2)
#     print(f'Found {name} at {cords} with confidence: {conf}')




#BUG: Training is computationally expensive
#results = model.train(data = load_dataset, epochs=2)
#metrics = model.val()
#print(metrics)


#TODO: to set up evaluation

