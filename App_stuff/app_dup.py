# Tkinter imports
from tkinter import filedialog
import tkinter as tk

# Matplotlib imports
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

# Image imports 
from PIL import Image, ImageTk
import cv2

# Technical imports
from sklearn.mixture import GaussianMixture as GMM
from math import pi
import numpy as np

# Misc imports
from copy import copy

""" DEFINE CONSTANTS """

# COLOURS
TEAL = '#0d7369'
WHITE = '#ffffff'
YELLOW = '#fafa57'

# FONTS
FUTURA80 = ('Futura', 80)
FUTURA50 = ('Futura', 50)

old_x = None
old_y = None

""" SET UP MAIN SCREEN """

# Set root
root = tk.Tk()

# Make main window full-screen
#root.attributes('-fullscreen', True)

# Set scaling factor for retina display
root.tk.call('tk', 'scaling', 2.0)

# Find geometry
width = root.winfo_screenwidth()
height= root.winfo_screenheight()

# Setting tkinter window size
root.geometry("%dx%d" % (width, height))
root.geometry('-2+0')
root.title("Draw My Horsey")

# Draw main screen
canvas = tk.Canvas(root, height=height, width=width, bg=TEAL)

# Configure canvas to occupy the whole main window
canvas.pack(fill=tk.BOTH, expand=True)

""" USEFUL FUNCTIONS """
def clear_canvas(canvas):
    """
    Clears all widgets in a given canvas
    """
    for widget in canvas.place_slaves():
        widget.place_forget()

""" TECHNICAL FUNCTIONS """

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
    canvas = FigureCanvasTkAgg(fig)
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

def get_simple_outline(bw_image):
    """
    Gets the simple outline edges
    """
    edges = cv2.Canny(bw_image, 80, 255)
    return edges

def get_complex_outline(image):
    """
    Gets the complex outline edges
    """
    edges = cv2.Canny(image, 60, 255)
    return edges

def find_optimum(pixels,trials):
    s1 = trials
    scores = [0] * s1

    for i in range(1, s1):
        gmm, labels = do_gmm(i, pixels)
        scores[i] = gmm.score(pixels) * -1

    d1 = np.zeros((s1))
    d2 = np.zeros((s1))
    strength = np.zeros((s1))
    rstr = np.zeros((s1))

    for i in range(s1):
        if i > 0:
            d1[i] = scores[i - 1] - scores[i]
            if i > 1:
                d2[i] = d1[i - 1] - d1[i]
                strength[i - 1] = d2[i] - d1[i]
                rstr[i-1] = strength[i-1]/(i-1)

    #opt = np.where(rstr == np.max(rstr))
    opt = np.where(strength == np.max(strength))
    return opt

""" Main function """

def gather_ellipses(path):
    """
    Main function, input the path to the image you want to find ellipses for 
    """
    # Read in image from path
    image = cv2.imread(path)
    im_dim = image.shape
    # Convert to black and white
    white_horse = colour2bw(image)
    # Get dimensions
    image_dims = white_horse.shape
    # Convert to pixels
    pixels = image2pixels(white_horse)
    # Find optimum num_components
    opt = find_optimum(pixels,9)
    # Do gmm
    gmm, labels = do_gmm(opt[0][0], pixels)
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
    simple_outline = get_simple_outline(white_horse)
    complex_outline = get_complex_outline(image)
    return im_dim, finished_ellipses, simple_outline, complex_outline


""" ELLIPSE/OUTLINE SCREEN """



def plot_simple_outline(ax, simple_outline):
    ax.imshow(simple_outline, cmap='binary', aspect='auto')
    return ax

def plot_complex_outline(ax, complex_outline):
    ax.imshow(complex_outline, cmap='binary', aspect='auto')
    return ax

def plot_ellipses(ax, ellipses):
    for ellipse in ellipses:
        ellipse.center = (ellipse.center[0], -ellipse.center[1])
        ellipse.angle = -ellipse.angle
        ellipse.set_fill(False)
        ellipse.set_linewidth(0.3)
        patch_cpy = copy(ellipse)
        # cut the umbilical cord the hard way
        patch_cpy.axes = None
        patch_cpy.figure = None
        patch_cpy.set_transform(ax.transData)
        ax.add_patch(patch_cpy)
    return ax

def plot_solo_ellipses(ax, im_dim, ellipses):
    white_bg = np.zeros(im_dim,dtype=np.uint8)
    white_bg.fill(255)
    ax.imshow(white_bg, cmap='binary', aspect='auto')
    ax = plot_ellipses(ax, ellipses)
    return ax

def make_image(im_dim=None, ellipses=None,simple_outline=None,complex_outline=None):

    # Create figure to show image on 
    image_fig = Figure(figsize=(4.2,4.2))
    image_subplot = image_fig.add_subplot(1,1,1)

    # Plot whichever has been selected 
    if simple_outline is not None:
        image_subplot = plot_simple_outline(image_subplot, simple_outline)
    if complex_outline is not None:
        image_subplot = plot_complex_outline(image_subplot, complex_outline)
    if ellipses is not None:
        image_subplot = plot_solo_ellipses(image_subplot, im_dim, ellipses)
    if ellipses is None and simple_outline is None and complex_outline is None:
        white_bg = np.zeros(im_dim,dtype=np.uint8)
        white_bg.fill(255)
        image_subplot.imshow(white_bg, cmap='binary', aspect='auto')
    
    image_subplot.axis('off')

    # Draw canvas
    c4nvas = FigureCanvasTkAgg(image_fig, canvas)
    c4nvas.draw()

    # Make image
    image_from_plot = np.frombuffer(image_fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(image_fig.canvas.get_width_height()[::-1] + (3,))
    image = Image.fromarray(image_from_plot)
    image = ImageTk.PhotoImage(image)

    return image

def better_paint(canvas, event):
    global old_x, old_y
    if old_x and old_y:
        canvas.create_line(old_x,old_y,event.x,event.y,width=2,fill='black',smooth=True)

    old_x = event.x
    old_y = event.y

def reset(event):    #reseting or cleaning the canvas 
    global old_x, old_y
    old_x = None
    old_y = None   

def paint(canvas, event):
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-1), (event.y-1)
    x2, y2 = (event.x+1), (event.y+1)
    color = "black"
    # display the mouse movement inside canvas
    canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)
    
def place_image(canvas, image):
    #canvas.delete("image")
    canvas.create_image(210, 223, image=image, tags="image", anchor='center')

def black_pix_white(image):
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    black_pixels = np.where(
        (image[:, :, 0] == 0) & 
        (image[:, :, 1] == 0) & 
        (image[:, :, 2] == 0)
    )

    # set those pixels to white
    image[black_pixels] = [255, 255, 255]
    return image

def white_pix_teal(image):
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    white_pixels = np.where(
        (image[:, :, 0] == 255) & 
        (image[:, :, 1] == 255) & 
        (image[:, :, 2] == 255)
    )

    # set those pixels to white
    image[white_pixels] = [13, 115, 105] 
    return image
    

def horse_image_w_maker(canvas, image_path, orig_image=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = black_pix_white(image)
    image = white_pix_teal(image)


    # Clear current canvas
    clear_canvas(canvas)

    # Create figure to show image on 
    image_fig = Figure(figsize=(3,3))
    image_subplot = image_fig.add_subplot(1,1,1)

    # Plot image
    image_subplot.imshow(image, aspect='auto')
    image_subplot.axis('off')

    # Draw canvas
    c4nvas = FigureCanvasTkAgg(image_fig, canvas)
    c4nvas.draw()

    # Make image
    image_from_plot = np.frombuffer(image_fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(image_fig.canvas.get_width_height()[::-1] + (3,))
    image = Image.fromarray(image_from_plot)

    if orig_image:
        return c4nvas
    else:
        return image


def drawing_screen(canvas, image_path):
    """
    Screen for drawing 
    """
    # Do all the gmm checking ellipses techincal stuff
    im_dim, finished_ellipses, simple_outline, complex_outline = gather_ellipses(image_path)

    # Clear current canvas
    clear_canvas(canvas)

    # Make each of the possible images from matplotlib
    blank = make_image(im_dim=im_dim)
    ellipses = make_image(im_dim, ellipses=finished_ellipses)
    simple = make_image(im_dim, simple_outline=simple_outline)
    compl3x = make_image(im_dim, complex_outline=complex_outline)

    # Make image one
    image_c4nvas = horse_image_w_maker(canvas, image_path, True)
    image_c4nvas.get_tk_widget().place(relx=0.35,rely=0.4, anchor="center")

    image_canvas = tk.Canvas(root, height = 446, width = 416, bg='white')
    image_canvas.bind('<B1-Motion>', lambda event: better_paint(image_canvas,event))
    image_canvas.bind('<ButtonRelease-1>', reset)
    image_canvas.place(relx=0.65,rely=0.4, anchor="center")

    place_image(image_canvas, blank)

    # Make check boxes for what to show
    link_var = tk.IntVar()           # variable to link the radiobuttons together

    
    show_blank = tk.Radiobutton(canvas, text="Show Nothing", variable=link_var, value=0, command=lambda : place_image(image_canvas, blank))
    show_ellipses = tk.Radiobutton(canvas, text="Show Ellipses", variable=link_var, value=1, command=lambda : place_image(image_canvas, ellipses))
    show_simple = tk.Radiobutton(canvas, text="Show Simple Outline", variable=link_var, value=2, command=lambda : place_image(image_canvas, simple))
    show_complex = tk.Radiobutton(canvas, text="Show Complex Outline", variable=link_var, value=3, command=lambda : place_image(image_canvas, compl3x))

    show_blank.place(relx=0.36,rely=0.7115, anchor="center")
    show_ellipses.place(relx=0.435,rely=0.7115, anchor="center")
    show_simple.place(relx=0.525,rely=0.7115, anchor="center")
    show_complex.place(relx=0.633,rely=0.7115, anchor="center")

    start_again_button = tk.Button(
    canvas,
    text='PICK ANOTHER',
    font=FUTURA50,
    fg=TEAL,
    bg=TEAL,
    activeforeground=YELLOW,
    command=lambda : image_picking(canvas, image_canvas)
    )
    start_again_button.place(relx=0.5,rely=0.85, anchor="center")

""" IMAGE PICKING SCREEN """

def image_picking(canvas, old_drawing=None):

    # Get image
    image_path = filedialog.askopenfilename()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = black_pix_white(image)
    image = white_pix_teal(image)

    if old_drawing is not None:
        old_drawing.place_forget()

    # Clear current canvas
    clear_canvas(canvas)

    # Create figure to show image on 
    image_fig = Figure(figsize=(3,3))
    image_subplot = image_fig.add_subplot(1,1,1)

    # Plot image
    image_subplot.imshow(image, aspect='auto')
    image_subplot.axis('off')

    # Draw canvas
    c4nvas = FigureCanvasTkAgg(image_fig, canvas)
    c4nvas.draw()

    # Place canvas
    c4nvas.get_tk_widget().place(relx=0.5,rely=0.4, anchor="center")

    # Make button to move to next step
    next_button = tk.Button(
        canvas,
        text='START DRAWING',
        font=FUTURA50,
        fg=TEAL,
        bg=TEAL,
        activeforeground=YELLOW,
        command=lambda : drawing_screen(canvas, image_path)
    )
    next_button.place(relx=0.5,rely=0.7, anchor="center")


""" START SCREEN """

# Labels
program_title = tk.Label(
    canvas,
    text='DRAW MY HORSEY!',
    font=FUTURA80,
    fg=WHITE,
    bg=TEAL
)
program_title.place(relx=0.5,rely=0.37, anchor="center")

start_button = tk.Button(
    canvas,
    text='BEGIN',
    font=FUTURA50,
    fg=TEAL,
    bg=TEAL,
    activeforeground=YELLOW,
    command=lambda : image_picking(canvas)
)
start_button.place(relx=0.5,rely=0.6, anchor="center")

# Start main loop
root.mainloop()