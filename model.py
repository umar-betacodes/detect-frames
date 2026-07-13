from ultralytics import YOLO

# Load your best trained weights
model = YOLO("yolo26m.pt") 

# Export to ONNX with dynamic axes
model.export(
    format="onnx", 
    dynamic=True,   # Unlocks dynamic batch size (N) and image dimensions
    simplify=True,  # Runs onnxslim to optimize the computational graph
    batch=16,       # Set to the max number of streams you expect per node
    opset=17        # Opset 13 is highly stable for edge deployment
)