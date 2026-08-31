import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10,1)

    def forward(self, x):
        return self.linear(x)

# 모델 객체 생성 (원본 모델)
model = SimpleModel()

# 방법1: 모델 전체 저장 및 불러오기
torch.save(model, "model_complete.pth")

loaded_model = torch.load(
    "model_complete.pth",
    weights_only=False
)
loaded_model.eval()  # 평가 모드로 전환


# 방법2: 모델의 가중치만 저장 및 불러오기
torch.save(model.state_dict(), "model_weights.pth")

weights_model = SimpleModel()  # 새로운 모델 객체 생성

state_dict = torch.load(
    "model_weights.pth",
    weights_only=True
)
weights_model.load_state_dict(state_dict)
weights_model.eval()  # 평가 모드로 전환


# 테스트 입력: 10개의 특성을 가진 샘플 1개
test_input = torch.randn(1, 10)

# 모델 예측 수행
with torch.inference_mode():
    origin_output = model(test_input)
    complete_output = loaded_model(test_input)
    weights_output = weights_model(test_input)

print("Origin Model Output:", origin_output)
print("Loaded Model Output (Weights False):", complete_output)
print("Loaded Model Output (Weights Only):", weights_output)

print("\n=== 두 결과가 같은지 확인 ===")
print(torch.allclose(complete_output, weights_output))