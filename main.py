# main.py

import tkinter as tk
from tkinter import ttk
from gui import DetectionGUI

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Real-Time Object Detection")
    root.geometry("1200x800")
    app = DetectionGUI(root)
    root.mainloop()
