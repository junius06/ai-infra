import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# GPU 사용 가능 여부에 따라 실행 장치 선택
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


# 모델 객체 생성
model = SimpleModel()

# 모델을 CPU 또는 GPU로 이동
model = model.to(device)

# 추론 모드로 전환
model.eval()


# 특성 10개를 가진 샘플 1개 생성
input_tensor = torch.randn(1, 10)

# 입력 데이터도 모델과 동일한 장치로 이동
input_tensor = input_tensor.to(device)


# 모델 추론
with torch.inference_mode():
    output = model(input_tensor)


# NumPy 변환을 위해 CPU로 이동
output = output.cpu().numpy()

print("Input shape:", input_tensor.shape)
print("Output:", output)