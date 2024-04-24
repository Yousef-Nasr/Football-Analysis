from ultralytics import YOLO

model = YOLO('models\\best.pt')

result = model.predict('input_video/goal2.mp4', save = True, device= 0) # device 0 to run on GPU
print(result[0])
print("*"*80)

for box in result[0].boxes:
    print(box)


