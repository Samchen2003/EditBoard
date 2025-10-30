import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_optical_flow(image1, image2):
    """
    Compute the optical flow between two images using Farneback method.

    Parameters:
    image1 (np.array): The first input image.
    image2 (np.array): The second input image.

    Returns:
    np.array: The computed optical flow.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def apply_optical_flow(image, flow):
    """
    Apply the optical flow to an image.

    Parameters:
    image (np.array): The input image.
    flow (np.array): The computed optical flow.

    Returns:
    np.array: The resulting image after applying the optical flow.
    """
    h, w = flow.shape[:2]
    # Generate the grid of coordinates and convert to float32
    flow_map = np.meshgrid(np.arange(w), np.arange(h))
    flow_map = np.stack(flow_map, axis=-1).astype(np.float32)
    
    # Add flow to coordinates
    flow_map -= flow

    # Warp the image using the flow map
    warped_image = cv2.remap(image, flow_map, None, cv2.INTER_LINEAR)

    return warped_image




def draw_flow(img, flow, step=16):
    """
    Draw optical flow vectors on the image.

    Parameters:
    img (np.array): The input image.
    flow (np.array): The optical flow.
    step (int): The step size for sampling the flow vectors.

    Returns:
    np.array: The image with flow vectors drawn.
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    # Create an image with flow vectors
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    # Draw end points
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis




def visualize_image_difference(image1, image2):
    """
    Visualize the difference between two images.

    Parameters:
    image1 (np.array): The first input image.
    image2 (np.array): The second input image.

    Returns:
    np.array: The image showing the differences.
    """
    # Ensure both images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions")

    # Compute the absolute difference between the two images
    diff = cv2.absdiff(image1, image2)

    # Convert the difference to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply a color map to the grayscale difference image to visualize it
    diff_colormap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

    return diff_colormap

def display_image(image, title='Image'):
    """
    Display an image using Matplotlib.

    Parameters:
    image (np.array): The image to display.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
