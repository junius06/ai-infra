import tensorflow as tf
# import tensorflow as keras

model = tf.keras.Sequential([
	tf.keras.Input(shape=(10,)),
	tf.keras.layers.Dense(50, activation='relu'),
	tf.keras.layers.Dense(1)
])

# 사용 가능한 GPU 조회
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

# GPU 메모리 동적 할당 설정
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

input_data = tf.random.normal((1,10))

device = "/GPU:0" if gpus else "/CPU:0"
print(f"Using device: {device}")
	
# GPU에서 모델 추론
with tf.device(device):
	predictions = model(input_data, training=False)

print("Predictions: ", predictions.numpy())