run order is as follows

- get_array
- border
- all_dims
- plot-all

get_array turns the coordinates from the datafile into a 2D array of the horse. 
It creates points of interest (POI) at a specified resolution to reduce computing time, 
these points are shown on the second image

border finds the pixels on the edge of the horse

all_dims calculates the dimensions of the largest possible ellipse for all POI inside the horse.

plot_all creates ellipses for all points using ellipse2.m, the most important are chosen iteratively and 
added onto the image.