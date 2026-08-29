import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# 사전 학습된 ResNet18 모델 불러오기
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# 샘플 이미지 다운로드
url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Prepare Image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# 추론 (Inference)
with torch.no_grad():
    output = model(input_batch)

# Get prediction
_, predicted = torch.max(output, 1)
print(f"Predicted class: {predicted.item()}")