import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def extract_shape(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_shape = cv2.approxPolyDP(largest_contour, epsilon, True)

    shape = np.squeeze(approx_shape)
    return shape

def interpolate_shapes(shape1, shape2, t):
    n1, _ = shape1.shape
    n2, _ = shape2.shape
    if n1 < n2:
        shape1 = np.concatenate([shape1, [shape1[0]]])
    elif n2 < n1:
        shape2 = np.concatenate([shape2, [shape2[0]]])

    interpolated_shape = []

    # linear interpolation vertices
    for vertex1, vertex2 in zip(shape1, shape2):
        interpolated_vertex = (1 - t) * vertex1 + t * vertex2
        interpolated_shape.append(interpolated_vertex)

    return np.array(interpolated_shape)

image1_path = 'images/circle.png'
image2_path = 'images/star.png'

try:
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
except Exception as e:
    print(f"Error reading images: {e}")
    exit(1)

if image1 is None or image2 is None:
    print("Failed to load images.")
    exit(1)

shape1 = extract_shape(image1)
shape2 = extract_shape(image2)

if shape1 is None or shape2 is None:
    print("Failed to extract shapes.")
    exit(1)

num_frames = 10

fig, ax = plt.subplots()

def update(frame):
    ax.cla() 

    t = frame / num_frames 
    interpolated_shape = interpolate_shapes(shape1, shape2, t)

    ax.plot(interpolated_shape[:, 0], interpolated_shape[:, 1], 'b-')
    ax.fill(interpolated_shape[:, 0], interpolated_shape[:, 1], 'b', alpha=0.3)

    ax.plot(shape1[:, 0], shape1[:, 1], 'r-')
    ax.plot(shape2[:, 0], shape2[:, 1], 'g-')

    ax.axis('equal')
    ax.set_title(f'Interpolation at t = {t:.2f}')

animation = FuncAnimation(fig, update, frames=num_frames + 1, interval=200)

animation.save('shape_interpolation.gif', writer='pillow')

plt.show()
