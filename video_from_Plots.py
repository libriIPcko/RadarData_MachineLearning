import numpy as np
import matplotlib.pyplot as plt
import cv2

# Generate example data
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure and axes
fig, ax = plt.subplots()

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("output_video.mp4", fourcc, 30, (640, 480))

# Loop over the data and create plots
for i in range(len(x)):
    # Clear the axes
    ax.clear()

    # Plot the data
    ax.plot(x[:i], y1[:i], label="sin(x)")
    ax.plot(x[:i], y2[:i], label="cos(x)")

    # Add legend and labels
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Save the figure as a PNG image
    fig.savefig(f"figure/frame_{i:04d}.png")

    # Load the image and write it to the video file
    img = cv2.imread(f"figure/frame_{i:04d}.png")
    video_writer.write(img)

    # Print progress
    #print(f"Processed frame {i + 1} of {len(x)}")

# Release the video writer object
video_writer.release()
