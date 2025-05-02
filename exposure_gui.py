import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
import os
import numpy as np
from PIL import Image, ImageTk

from exposure_core import create_photo_avg, create_photo_grad, create_photo_blend, get_frames, create_photo_blend_with_pyramids, colorful_gradient_threshold


class ExposureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Long Exposure from Video")

        self.video_path = None
        self.result_img = None
        self.cap = None
        self.playing = False
        self.preview_resolution = (400, 300)

        # Hlavné rozloženie: 2 stĺpce (video / výstup)
        main_frame = ttk.Frame(root)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.video_panel = ttk.Label(main_frame)
        self.video_panel.grid(row=0, column=0, padx=5, pady=5)

        self.result_panel = ttk.Label(main_frame)
        self.result_panel.grid(row=0, column=1, padx=5, pady=5)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Ovládacie prvky
        control_frame = ttk.Frame(root)
        control_frame.pack(pady=10)

        # Výber metódy
        self.method = tk.StringVar(value="avg")
        ttk.Label(control_frame, text="Select method:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(control_frame, text="Average", variable=self.method, value="avg").grid(row=0, column=1)
        ttk.Radiobutton(control_frame, text="Gradient", variable=self.method, value="grad").grid(row=0, column=2)
        ttk.Radiobutton(control_frame, text="Blend", variable=self.method, value="blend").grid(row=0, column=3)
        ttk.Radiobutton(control_frame, text="Blend with pyramids", variable=self.method, value="blend_pyramids").grid(row=0, column=4)

        # Step slider
        self.step = tk.IntVar(value=1)
        ttk.Label(control_frame, text="Step:").grid(row=1, column=0, sticky="w")
        ttk.Scale(control_frame, from_=1, to=30, variable=self.step, orient='horizontal').grid(row=1, column=1, columnspan=2, sticky="we")

        # Alpha slider
        self.alpha = tk.DoubleVar(value=0.5)
        ttk.Label(control_frame, text="Colours (only for blend):").grid(row=2, column=0, sticky="w")
        ttk.Scale(control_frame, from_=0, to=1, variable=self.alpha, orient='horizontal').grid(row=2, column=1, columnspan=2, sticky="we")

        ttk.Button(control_frame, text="📂 Load Video", command=self.load_video).grid(row=4, column=4)


        # Tlačidlá
        ttk.Button(control_frame, text="▶️ Play", command=self.play_video).grid(row=4, column=0)
        ttk.Button(control_frame, text="⏸️ Pause", command=self.pause_video).grid(row=4, column=1)
        ttk.Button(control_frame, text="⚙️ Process", command=self.process_video).grid(row=4, column=2)
        ttk.Button(control_frame, text="💾 Save", command=self.save_image).grid(row=4, column=3)
        self.root.geometry("1400x800")


    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            messagebox.showinfo("Video loaded", f"Loaded:\n{path}")
            self.load_preview_frame()


    def drop_video(self, event):
        path = event.data.strip('{}')
        if path.endswith(('.mp4', '.avi', '.mov')):
            self.video_path = path
            messagebox.showinfo("Video loaded", f"Loaded:\n{path}")
            self.load_preview_frame()
        else:
            messagebox.showerror("Invalid file", "Please drop a valid video file.")

    def load_preview_frame(self):
        self.cap = cv2.VideoCapture(self.video_path)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame, self.video_panel)
        self.cap.release()

    def play_video(self):
        if not self.video_path:
            return
        self.playing = True
        self.cap = cv2.VideoCapture(self.video_path)
        self.play_loop()

    def play_loop(self):
        if not self.playing:
            if self.cap:
                self.cap.release()
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.playing = False
            return
        self.display_frame(frame, self.video_panel)
        self.root.after(30, self.play_loop)

    def pause_video(self):
        self.playing = False

    def display_frame(self, frame, label_widget):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize(self.preview_resolution)
        photo = ImageTk.PhotoImage(img)
        label_widget.config(image=photo)
        label_widget.image = photo  # prevent garbage collection

    def process_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video selected!")
            return

        frames = get_frames(self.video_path)
        method = self.method.get()
        step = self.step.get()
        alpha = self.alpha.get()

        if method == "avg":
            self.result_img = create_photo_avg(frames, short=False)
        elif method == "grad":
            self.result_img = create_photo_grad(frames, short=False, step_cmd=step)
        elif method == "blend":
            self.result_img = create_photo_blend(frames, short=False, step_cmd=step, alpha=alpha)
        elif method == "blend_pyramids":
            self.result_img = create_photo_blend_with_pyramids(frames, short=False, step_cmd=step, alpha=alpha)
        elif method == "colorful_gradient":
            self.result_img = colorful_gradient_threshold(frames, step=step)

        # Zobraz výstupný obrázok
        self.display_frame(self.result_img, self.result_panel)
        messagebox.showinfo("Done", "Processing complete!")

    def save_image(self):
        if self.result_img is None:
            messagebox.showerror("Error", "No image to save!")
            return
        out_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
        if out_path:
            cv2.imwrite(out_path, self.result_img)
            messagebox.showinfo("Saved", f"Image saved to:\n{out_path}")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ExposureApp(root)
    root.mainloop()
