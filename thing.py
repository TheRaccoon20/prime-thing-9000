import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def generate_and_normalize_dataset(start, end):
    data = []
    labels = []

    for num in range(start, end + 1):
        data.append([num])
        labels.append([1 if is_prime(num) else 0])

    data = np.array(data) / end  # Normalize data

    return data, np.array(labels)

start_num = 1
end_num = 2**127 - 1
x_train, y_train = generate_and_normalize_dataset(start_num, end_num)

# Add a normalization layer cuz why not
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(1,)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 100 epochs should be sufficient
model.fit(x_train, y_train, epochs=100, batch_size=8)

model.save('prime_model.h5')

def evaluate_input(num):
    input_data = np.array([[num]]) / end_num  # Normalize input with end_num
    prediction = model.predict(input_data)[0][0]

    if prediction > 0.5:
        return "Prime"
    else:
        return "Composite"

user_input = int(input("Enter number: "))
result = evaluate_input(user_input)

print(f"The number {user_input} is predicted to be: {result}")
