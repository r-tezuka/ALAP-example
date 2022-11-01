import tkinter as tk
import svgpathtools as svg
import numpy as np
CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 700


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ALAP example")
        self.geometry("1200x800")
        c = tk.Canvas(self, background="white",
                      width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        c.grid(row=0, column=0, sticky=tk.NSEW)
        self.start = None
        c.bind("<Motion>", self.move)
        c.bind("<ButtonPress>", self.button_press)
        c.bind("<ButtonRelease>", self.button_release)
        self.canvas = c
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.is_dragged = False
    
    def draw(self, xs, ys, handles):
        r = 3
        
        for i in range(len(xs)):
            color = 'black'
            if i in handles:
                color = 'red'
            self.canvas.create_oval(xs[i] - r, ys[i] - r, xs[i] + r, ys[i] + r, fill = color, outline='')


    def move(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

    def button_press(self, event):
        self.is_dragged = True

    def button_release(self, event):
        self.is_dragged = False

if __name__ == "__main__":
    app = App()
    app.mainloop()
