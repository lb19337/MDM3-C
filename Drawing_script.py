from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def colour2bw(image):
    grey_horse = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (_, white_horse) = cv2.threshold(grey_horse, 10, 255, cv2.THRESH_BINARY)
    return white_horse

def image2pixels(bw_image):    
    # Initialise arrays to store positions of each pixel
    x_pixels = np.zeros(len(bw_image.flatten()))
    y_pixels = np.zeros(len(bw_image.flatten()))
    # Store position of each pixel
    count = 0
    for row in range(bw_image.shape[0]):
        for column in range(bw_image.shape[1]):
            if bw_image[row,column] != 0:
                x_pixels[count] = column
                y_pixels[count] = -row
            count += 1
    # Remove datapoints where there are no pixels
    x_pixels = x_pixels[x_pixels != 0]
    y_pixels = y_pixels[y_pixels != 0]
    return x_pixels, y_pixels

def do_gmm(ncomponents, pixels):
    gmm = GMM(n_components=ncomponents, random_state=0)
    xy = np.transpose(np.array(pixels))
    labels = gmm.fit(xy).predict(xy)
    return gmm, labels

def ellipse_data(gmm):
    # Initialise lists for storing data
    positions = []
    widths = []
    heights = []
    angles = []
    # Get ellipse data
    for position, covariance in zip(gmm.means_, gmm.covariances_):
        positions.append(position)
        if covariance.shape == (2, 2):
            U, s, _ = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        widths.append(width)
        heights.append(height)
        angles.append(angle)
    data = {'positions':positions, 'widths':widths, 'heights':heights, 'angles':angles}
    return data

def plot_image(image, ellipses):
    plt.imshow(image, cmap='gray', aspect='auto')
    # Draw each Ellipse
    for i in range(len(ellipses['positions'])):
        nsig = 1.7
        plt.gca().add_patch(Ellipse(ellipses['positions'][i], 
                            nsig * ellipses['widths'][i], 
                            nsig * ellipses['heights'][i],
                            ellipses['angles'][i],
                            color='k',
                            fill=False))
    plt.show()

def main(path):
    # Read in image from path
    image = cv2.imread(path)
    # Convert to black and white
    white_horse = colour2bw(image)
    # Convert to pixels
    pixels = image2pixels(white_horse)
    # Do gmm
    gmm, labels = do_gmm(6, pixels)
    # Get ellipses data
    ellipses = ellipse_data(gmm)
    # Plot black and white image with ellipses on top
    plot_image(white_horse, ellipses)

if __name__ == "__main__":
    main("Images/horse.png")

        