# gui.py

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import time
from detector import ObjectDetector
from tracker import ObjectTracker

class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.running = False
        self.fps_log = []
        self.video_thread = None

        self.start_screen()

    def start_screen(self):
        self.clear_window()
        self.start_button = ttk.Button(self.root, text="Start", command=self.start_detection)
        self.start_button.pack(expand=True)

    def start_detection(self):
        self.clear_window()
        self.running = True

        # Layout: 3 columns
        self.left_frame = tk.Frame(self.root, width=300, bg="lightgreen")
        self.middle_frame = tk.Frame(self.root, width=600)
        self.right_frame = tk.Frame(self.root, width=300, bg="lightcoral")

        self.left_frame.pack(side="left", fill="y")
        self.middle_frame.pack(side="left", expand=True)
        self.right_frame.pack(side="right", fill="y")

        tk.Label(self.left_frame, text="Recently Added", bg="lightgreen").pack()
        tk.Label(self.right_frame, text="Recently Removed", bg="lightcoral").pack()

        # Video feed label
        self.video_label = tk.Label(self.middle_frame)
        self.video_label.pack()

        self.fps_label = tk.Label(self.middle_frame, text="FPS: 0.00")
        self.fps_label.pack()

        self.stop_button = ttk.Button(self.middle_frame, text="Stop", command=self.stop_detection)
        self.stop_button.pack(pady=10)

        # Start camera in new thread
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.start()

    def video_loop(self):
        cap = cv2.VideoCapture(0)
        prev_time = time.time()
    
        detector = ObjectDetector()
        tracker = ObjectTracker()
    
        tracked_ids = set()
        added = []
        removed = []
    
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = detector.detect(frame)
            tracks = tracker.update(frame, detections)
    
            current_ids = set([t['track_id'] for t in tracks])
    
            # Recently added
            new_ids = current_ids - tracked_ids
            for tid in new_ids:
                for t in tracks:
                    if t['track_id'] == tid:
                        added.append((tid, t['class_name']))
                        tk.Label(self.left_frame, text=f"ID {tid} - {t['class_name']}", bg="lightgreen").pack()
    
            # Recently removed
            lost_ids = tracked_ids - current_ids
            for tid in lost_ids:
                name = next((item[1] for item in added if item[0] == tid), "Unknown")
                removed.append((tid, name))
                tk.Label(self.right_frame, text=f"ID {tid} - {name}", bg="lightcoral").pack()
    
            tracked_ids = current_ids
    
            # Draw on frame
            for t in tracks:
                x1, y1, x2, y2 = map(int, t['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{t['class_name']} ID:{t['track_id']}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            self.fps_log.append(fps)
            self.fps_label.config(text=f"FPS: {fps:.2f}")
    
            # Convert and show in GUI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
    
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
    
        cap.release()

    def stop_detection(self):
        self.running = False
        self.video_thread.join()
        self.show_summary()

    def show_summary(self):
        self.clear_window()

        ttk.Label(self.root, text="Detection Stopped").pack(pady=10)

        ttk.Button(self.root, text="Download Detection Log", command=self.download_detection_log).pack(pady=5)
        ttk.Button(self.root, text="Download Removal Log", command=self.download_removal_log).pack(pady=5)
        ttk.Button(self.root, text="Show FPS Graph", command=self.plot_fps).pack(pady=5)

    def download_detection_log(self):
        with open("output/logs/detections_log.txt", "w") as f:
            f.write("Sample Detection Log\n")  # Replace with real log
        filedialog.asksaveasfilename(defaultextension=".txt")

    def download_removal_log(self):
        with open("output/logs/deletions_log.txt", "w") as f:
            f.write("Sample Removal Log\n")  # Replace with real log
        filedialog.asksaveasfilename(defaultextension=".txt")

    def plot_fps(self):
        import matplotlib.pyplot as plt
        plt.plot(self.fps_log)
        plt.title("FPS Over Time")
        plt.xlabel("Frames")
        plt.ylabel("FPS")
        plt.grid(True)
        plt.show()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
