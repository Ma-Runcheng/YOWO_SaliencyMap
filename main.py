import os
import tkinter as tk

import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
import drawCAM

class ImageViewer:
    def __init__(self):
        self.images = []
        self.origin_images = []
        self.current_index = 0

        self.window = tk.Tk()
        self.window.title("Image Viewer")

        self.frame = tk.Frame(self.window)
        self.frame.pack()

        self.label = tk.Label(self.frame)
        self.label.pack()

        menubar = tk.Menu(self.window)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Folder", command=self.load_images_from_folder)
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="原图", command=self.show_original_image)
        menubar.add_cascade(label="动作检测检测", command=self.show_detection_image)
        menubar.add_cascade(label="yolo层显著图", command=self.show_YOLO_Grad)
        menubar.add_cascade(label="Attention层显著图", command=self.show_Attention_Grad)

        self.window.config(menu=menubar)

        self.window.bind('<KeyPress>', self.key_pressed)

    def load_images_from_folder(self):
        self.images = []
        self.origin_images = []
        foderName = self.getFolderName()
        for filename in os.listdir(foderName):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                img = Image.open(os.path.join(foderName, filename))
                self.origin_images.append(img)
        self.images = self.origin_images
        self.show_image()

    def show_image(self):
        img = self.images[self.current_index]
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.label.config(image=photo)
        self.label.image = photo

    def key_pressed(self, event):
        key = event.char.lower()
        if key == 'a':
            self.current_index = (self.current_index - 1) % len(self.images)
            self.show_image()
        elif key == 'd':
            self.current_index = (self.current_index + 1) % len(self.images)
            self.show_image()

    def run(self):
        self.window.mainloop()

    def getFolderName(self):
        forderName = filedialog.askdirectory()
        return forderName

    def show_original_image(self):
        self.images = self.origin_images
        self.show_image()

    def show_detection_image(self):
        self.images = self.origin_images
        image = drawCAM.get_Detection_Image(self.images)
        image = [Image.fromarray(img) for img in image]
        self.images = image
        self.show_image()

    def show_YOLO_Grad(self):
        self.images = self.origin_images
        gradMap = drawCAM.get_YOLO_Grad_Map(self.images)
        gradMap = [Image.fromarray(grad) for grad in gradMap]
        self.images = gradMap
        self.show_image()

    def show_Attention_Grad(self):
        self.images = self.origin_images
        gradMap = drawCAM.get_Attention_Grad_Map(self.images)
        gradMap = [Image.fromarray(grad) for grad in gradMap]
        self.images = gradMap
        self.show_image()


# 创建图片浏览器对象，传入图片文件夹路径
image_viewer = ImageViewer()
image_viewer.run()
