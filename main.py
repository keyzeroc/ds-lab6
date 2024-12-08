import numpy as np


# Функція активації (сигмоїда)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Похідна функції активації
def sigmoid_derivative(x):
    return x * (1 - x)


# Вхідні дані
X = np.array([[0, 1, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]])

# Вихідні дані
y = np.array([[0], [0], [1], [1], [1]])

# Ініціалізація випадкових ваг
np.random.seed(1)
weights = 2 * np.random.random((3, 1)) - 1

# Навчання нейронної мережі
for iteration in range(10000):
    input_layer = X
    outputs = sigmoid(np.dot(input_layer, weights))
    error = y - outputs
    adjustments = error * sigmoid_derivative(outputs)
    weights += np.dot(input_layer.T, adjustments)

# Виведення результатів
print("Вихідні дані після навчання:")
print(outputs)

# Виведення ваг після навчання
print("Ваги після навчання:")
print(weights)
