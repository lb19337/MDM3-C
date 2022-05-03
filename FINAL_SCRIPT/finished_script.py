from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from math import pi
from copy import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np

""" Utility functions """

def remove_duplicates(l1st):
    """
    Removes duplicates from list and returns new list
    """
    return list(dict.fromkeys(l1st))




""" Converting image to datapoints """

def colour2bw(image):
    """ 
    Converts a colour image to black and white
    """
    grey_horse = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (_, white_horse) = cv2.threshold(grey_horse, 10, 255, cv2.THRESH_BINARY)
    return white_horse

def image2pixels(bw_image):
    """
    Converts an image to x,y, datapoints for each pixel
    """   
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
    # Make as array for use with 
    pixels = np.transpose(np.array((x_pixels,y_pixels)))
    return pixels






""" Doing GMM """

def do_gmm(ncomponents, pixels):
    """
    Fits a gmm model to data and returns this model along with the labels
    """
    gmm = GMM(n_components=ncomponents, random_state=0)
    labels = gmm.fit(pixels).predict(pixels)
    return gmm, labels





""" Ellipse gathering and checking """

def ellipse_data(gmm):
    """
    Get the ellipse dimension data from the gmm model
    """
    # Initialise lists for storing data
    positions = []
    widths = []
    heights = []
    angles = []
    areas = []
    # Get ellipse data
    for position, covariance in zip(gmm.means_, gmm.covariances_):
        position[1] = position[1]
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
        areas.append(pi*width*angle)
    data = {'positions':positions, 'widths':widths, 'heights':heights, 'angles':angles, 'areas':areas}
    return data

def get_cluster(pixels, labels, cluster_label):
    """
    Gets points from a specific cluster
    """
    idx = list(np.asarray(labels==cluster_label).nonzero()[0])
    cluster_pixels = pixels[idx,:]
    return cluster_pixels

def get_in_points(ellipses, pixels, labels, ellipse_id):
    """
    Get the datapoints that are in the given ellipse
    """
    # Get ellipse
    nsig = 1.7
    ellipse = Ellipse(ellipses['positions'][ellipse_id], 
                            nsig * ellipses['widths'][ellipse_id], 
                            nsig * ellipses['heights'][ellipse_id],
                            ellipses['angles'][ellipse_id],
                            color='k',
                            fill=True)
    # Define indexes for points that are in ellipse
    in_datapoints = np.empty((1,2))
    # Loop through clusters and see h
    for cluster_id in range(len(ellipses["positions"])): 
        cluster_datapoints = get_cluster(pixels, labels, cluster_id)
        in_idx = ellipse.contains_points(cluster_datapoints)
        in_datapoints = np.concatenate((in_datapoints, cluster_datapoints[in_idx]))
    # Remove initialised first point
    in_datapoints = in_datapoints[1:,:]
    return ellipse, in_datapoints

def image_from_plot(ellipse, datapoints, image_dims):
    """
    Makes plot and converts it to an image array
    """
    # Make plotting objects
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    # Add patch and scatter points to show blackspace
    ax.add_patch(ellipse)
    ax.scatter(datapoints[:,0], datapoints[:,1], s=0.5, c='w')
    ax.set_xlim([0, 2*image_dims[0]])
    ax.set_ylim([-image_dims[1], 0])
    # Remove axis and margins
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    # Draw canvas
    canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = Image.fromarray(image_from_plot)
    return image

def rgba_to_grey(image):
    """
    Converts rgba image to greyscale
    """
    return image.convert('LA')

def count_whitespace(image):
    """
    Counts amount of whitespace in pixels
    """
    image_array = np.array(image)
    num_white_pixels = np.sum(image_array == 0)
    return num_white_pixels

def check_whitespace(ellipses, pixels, labels, ellipse_id, image_dims):
    """
    Check the amount of white space 
    """
    # Get the index of all the points inside the ellipse
    ellipse, in_points = get_in_points(ellipses, pixels, labels, ellipse_id)
    # Plot the ellipse and all the datapoints inside the ellipse and converts it to image array
    colour_image = image_from_plot(ellipse, in_points, image_dims)
    # Convert image array from rgba to grayscale
    bw_image = rgba_to_grey(colour_image)
    # Count amount of whitespace
    whitespace = count_whitespace(bw_image)
    return ellipse, whitespace

def check_ellipses(ellipses, pixels, labels, image_dims):
    """ 
    Checks ellipses by how much whitespace they contain
    """
    num_ellipses = len(ellipses["positions"])
    final_ellipses = []
    faulty_clusters = []
    for ellipse_id in range(num_ellipses):
        ellipse, whitespace = check_whitespace(ellipses, pixels, labels, ellipse_id, image_dims)
        if whitespace < 100:
            final_ellipses.append(ellipse)
        else:
            faulty_clusters.append(get_cluster(pixels, labels, ellipse_id))
    return final_ellipses, faulty_clusters





""" Function for iterative looping """
def gmm_iter(pixels):
    # do gmm
    gmm, labels = do_gmm(2, pixels)
    # get ellipses
    ellipses = ellipse_data(gmm)
    # check ellipses
    good_ellipses, faulty_clusters = check_ellipses(ellipses, pixels, labels)



""" Plotting finished product """
def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def plot_image(image, finished_ellipses):
    """
    Plot the final image with the ellipses drawn on
    """
    clear_plt()
    ax = plt.gca()
    ax.scatter(image[:,0], image[:,1], s=0.5, c='b')
    for ellipse in finished_ellipses:
        ellipse.set_fill(False)
        patch_cpy = copy(ellipse)
        # cut the umbilical cord the hard way
        patch_cpy.axes = None
        patch_cpy.figure = None
        patch_cpy.set_transform(ax.transData)
        ax.add_patch(patch_cpy)
    plt.show()





""" Main function """

def main(path):
    """
    Main function, input the path to the image you want to find ellipses for 
    """
    # Read in image from path
    image = cv2.imread(path)
    # Convert to black and white
    white_horse = colour2bw(image)
    # Get dimensions
    image_dims = white_horse.shape
    # Convert to pixels
    pixels = image2pixels(white_horse)
    # Do gmm
    gmm, labels = do_gmm(6, pixels)
    # Get ellipses data
    ellipses = ellipse_data(gmm)
    # Loop through ellipses and check that they fit
    good_ellipses, faulty_clusters = check_ellipses(ellipses, pixels, labels, image_dims)
    finished_ellipses = good_ellipses
    while len(faulty_clusters) > 0:
        f_clusters = []
        if len(faulty_clusters) > 100:
            break
        # Loop through faulty clusters
        for cluster in faulty_clusters:
            gmm, labels = do_gmm(2, cluster)
            # Get ellipses data
            ellipses = ellipse_data(gmm)
            # Check ellipses
            g_ellipses, f_cluster = check_ellipses(ellipses, cluster, labels, image_dims)
            # Append faulty clusters to list
            f_clusters += f_cluster
            # Append good ellipses to list
            finished_ellipses += g_ellipses
        # Change list of faulty clusters
        faulty_clusters = f_clusters

    # Plot black and white image with ellipses on top
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    plot_image(pixels, finished_ellipses)

if __name__ == "__main__":
    main("Images/horse.png")