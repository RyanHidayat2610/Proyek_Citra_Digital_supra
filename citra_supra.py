import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengolahan Citra Digital SUPRA")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f4f3")
        self.root.minsize(1000, 600)

        self.image = None
        self.processed_image = None

        main_frame = tk.Frame(self.root, bg="#f0f4f3")
        main_frame.pack(fill="both", expand=True)

        sidebar = tk.Frame(main_frame, bg="#14532d", width=220)
        sidebar.pack(side="left", fill="y")

        title = tk.Label(sidebar, text="Image Tools", bg="#14532d", fg="white", font=("Segoe UI", 16, "bold"))
        title.pack(pady=(20, 10))

        self.create_button(sidebar, "📂 Input Gambar", self.load_image)
        self.create_button(sidebar, "🌑 Grayscale", self.to_grayscale)
        self.create_button(sidebar, "🔳 Biner", self.to_binary)
        self.create_button(sidebar, "💡 Brightness", self.adjust_brightness)
        self.create_button(sidebar, "🚫 Logika NOT", self.logic_not)
        self.create_button(sidebar, "📊 Histogram", self.show_histogram_all)
        self.create_button(sidebar, "⚡ Edge Detection", self.edge_detection_menu)
        self.create_button(sidebar, "➖ Erosi", self.morph_erode_custom)

        content = tk.Frame(main_frame, bg="#ffffff")
        content.pack(side="right", fill="both", expand=True)

        image_frame = tk.Frame(content, bg="#ffffff")
        image_frame.pack(fill="both", expand=True, padx=10, pady=10)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(1, weight=1)

        self.original_image_label = tk.Label(image_frame, bg="#ffffff", text="Gambar Asli")
        self.original_image_label.grid(row=0, column=0, padx=10, pady=5)

        self.processed_image_label = tk.Label(image_frame, bg="#ffffff", text="Gambar Hasil")
        self.processed_image_label.grid(row=0, column=1, padx=10, pady=5)

        self.original_canvas = tk.Label(image_frame, bg="#dddddd")
        self.original_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.processed_canvas = tk.Label(image_frame, bg="#dddddd")
        self.processed_canvas.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

    def create_button(self, parent, text, command):
        btn = tk.Button(parent, text=text, command=command, font=("Segoe UI", 10, "bold"),
                        bg="#22c55e", fg="white", activebackground="#16a34a",
                        relief="flat", width=20, height=2)
        btn.pack(pady=2)
        return btn

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif")])
        if file_path:
            self.image = Image.open(file_path)
            self.display_image(self.image, self.original_canvas)
            self.processed_image = None
            self.processed_canvas.config(image='')

    def display_image(self, pil_img, target_label):
        max_size = (900, 800)
        pil_img = pil_img.copy()
        pil_img.thumbnail(max_size)
        img_tk = ImageTk.PhotoImage(pil_img)
        target_label.img = img_tk
        target_label.config(image=img_tk)

    def to_grayscale(self):
        if self.image:
            img = np.array(self.image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.processed_image = Image.fromarray(gray)
            self.display_image(self.processed_image, self.processed_canvas)

    def to_binary(self):
        if self.image:
            img = np.array(self.image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.processed_image = Image.fromarray(binary)
            self.display_image(self.processed_image, self.processed_canvas)

    def adjust_brightness(self):
        if self.image:
            val = simpledialog.askinteger("Brightness", "Masukkan nilai brightness (0 hingga 255):", minvalue=0, maxvalue=255)
            if val is not None:
                img = np.array(self.image)
                result = cv2.convertScaleAbs(img, alpha=1, beta=val)
                self.processed_image = Image.fromarray(result)
                self.display_image(self.processed_image, self.processed_canvas)

    def logic_not(self):
        if self.image:
            img = np.array(self.image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            result = cv2.bitwise_not(gray)
            self.processed_image = Image.fromarray(result)
            self.display_image(self.processed_image, self.processed_canvas)

    def show_histogram_all(self):
        if self.image:
            img = np.array(self.image)
            plt.figure("Histogram RGB")
            for i, color in enumerate(['red', 'green', 'blue']):
                plt.hist(img[:, :, i].ravel(), bins=256, range=[0, 256], color=color, alpha=0.5, label=color.upper())
            plt.title("Histogram RGB")
            plt.xlabel("Intensitas")
            plt.ylabel("Jumlah Piksel")
            plt.legend()
            plt.grid(True)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            plt.figure("Histogram Grayscale")
            plt.hist(gray.ravel(), bins=256, range=[0, 256], color='black')
            plt.title("Histogram Grayscale")
            plt.xlabel("Intensitas")
            plt.ylabel("Jumlah Piksel")
            plt.grid(True)

            plt.show()

    def edge_detection_menu(self):
        if not self.image:
            return

        popup = tk.Toplevel(self.root)
        popup.title("Metode Deteksi Tepi (Sobel)")
        popup.geometry("300x150")
        popup.configure(bg="#f0f4f3")

        tk.Label(popup, text="Pilih metode deteksi tepi:", bg="#f0f4f3", font=("Segoe UI", 10)).pack(pady=10)

        def apply(method):
            img = np.array(self.image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if method == "SobelX":
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                edges = cv2.convertScaleAbs(sobelx)
            elif method == "SobelY":
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges = cv2.convertScaleAbs(sobely)

            self.processed_image = Image.fromarray(edges)
            self.display_image(self.processed_image, self.processed_canvas)
            popup.destroy()

        tk.Button(popup, text="Sobel Horizontal", width=25, command=lambda: apply("SobelX")).pack(pady=5)
        tk.Button(popup, text="Sobel Vertikal", width=25, command=lambda: apply("SobelY")).pack(pady=5)

    def morph_erode_custom(self):
        if not self.image:
            return

        se_choices = {
            "Vertikal": np.array([[0,1,0],[0,1,0],[0,1,0]], dtype=np.uint8),
            "Horizontal": np.array([[0,0,0],[1,1,1],[0,0,0]], dtype=np.uint8),
            "Kotak Penuh": np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8)
        }

        popup = tk.Toplevel(self.root)
        popup.title("Pilih Elemen Penstruktur")
        popup.geometry("300x150")
        popup.configure(bg="#f0f4f3")

        tk.Label(popup, text="Elemen Penstruktur:", bg="#f0f4f3", font=("Segoe UI", 10)).pack(pady=10)
        selected = tk.StringVar(popup)
        selected.set("Kotak Penuh")
        tk.OptionMenu(popup, selected, *se_choices.keys()).pack()

        def apply_erode():
            kernel = se_choices[selected.get()]
            img = np.array(self.image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            result = cv2.erode(binary, kernel, iterations=1)
            self.processed_image = Image.fromarray(result)
            self.display_image(self.processed_image, self.processed_canvas)
            popup.destroy()

        tk.Button(popup, text="Terapkan", command=apply_erode,
                  bg="#22c55e", fg="white", font=("Segoe UI", 10, "bold")).pack(pady=10)

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
