import tkinter as tk
import os

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
canvas = tk.Canvas(root, height=height, width=width, bg='#0d7369')

# Configure canvas to occupy the whole main window
canvas.pack(fill=tk.BOTH, expand=True)

# Labels
program_title = tk.Label(canvas, text='Welcome to draw my horsey!')
program_title.pack()

# Start main loop
root.mainloop()