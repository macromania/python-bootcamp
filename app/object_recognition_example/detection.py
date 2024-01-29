from imageai.Detection import ObjectDetection
import os


execution_path = os.getcwd()
print(f"Running from {execution_path}")

print("Loading model...")
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()

print("Model loaded!")

print("Detecting objects...")
detections = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path, "input", "image.jpg"), 
    output_image_path=os.path.join(execution_path, "output", "image_new.jpg")
)

print("Detected objects!")

for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])