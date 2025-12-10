import onnx
import numpy as np
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import onnxruntime as ort


model = onnx.load("hair_classifier_v1.onnx")
graph = model.graph

print("Inputs:")
for inp in graph.input:
    print(inp.name)

print("\nOutputs:")
for out in graph.output:
    print(out.name)

train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])

image = Image.open("yf_dokzqy3vcritme8ggnzqlvwa.jpeg")
image = train_transforms(image)
np_image = image.numpy()
first_pixel = np_image[0,0,0]
print(first_pixel)




session = ort.InferenceSession("hair_classifier_v1.onnx")

# Get input and output names
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Input name:", input_name)
print("Output name:", output_name)

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    x = train_transforms(img)          # tensor: (C, H, W)
    x = x.unsqueeze(0)           # add batch dim: (1, C, H, W)
    x = x.numpy().astype(np.float32)
    return x

x = preprocess_image("yf_dokzqy3vcritme8ggnzqlvwa.jpeg")

outputs = session.run([output_name], {input_name: x})
y = outputs[0]     # numpy array
print("Raw model output:", y)