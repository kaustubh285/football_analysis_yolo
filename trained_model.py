from ultralytics import YOLO


model = YOLO("models/medium/best.pt")
# results = model.predict("input/08fd33_4_short.mp4", save=True, project="runs/detect")
results = model.predict("input/dls_rec.mov", save=True, project="runs/detect")

print(results[0])
