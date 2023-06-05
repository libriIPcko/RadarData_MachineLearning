import matplotlib.pyplot as plt
from PIL import Image
import imageio

# generate some random data
x = range(10)
y = [i**2 for i in x]

# create a list of images
images = []
for i in range(len(x)):
    plt.plot(x[:i+1], y[:i+1])
    plt.title('Plot {}'.format(i+1))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.ylim(0, 80)
    plt.xlim(0, 10)
    # save the plot as an image
    fig = plt.gcf()
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    images.append(image)

# create the GIF from the list of images
imageio.mimsave('plots.gif', images, duration=0.5)