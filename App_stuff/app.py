from tkinter import filedialog, Text
import tkinter as tk
import os

# COLOURS
TEAL = '#0d7369'
WHITE = '#ffffff'
YELLOW = '#fafa57'

# FONTS
FUTURA80 = ('Futura', 80)
FUTURA50 = ('Futura', 50)

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
    activeforeground=YELLOW
)
start_button.place(relx=0.5,rely=0.6, anchor="center")

# Start main loop
root.mainloop()