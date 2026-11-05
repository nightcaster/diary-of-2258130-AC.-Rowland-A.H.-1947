import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageEnhance, ImageOps, ImageTk
import numpy as np
import cv2

# Point to your image directory
image_dir = r"C:\TEMP"

images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
if not images:
    print("No images found in the specified directory!")
    exit()

original_img = Image.open(os.path.join(image_dir, images[0])).convert('RGB')

class DiaryTunerApp:
    def __init__(self, root, img):
        self.root = root
        self.root.title("Diary Enhancement Tuner - Hotspot & Uniformity Studio")
        self.root.geometry("1650x950")
        
        self.original_img = img
        self.after_img_pil = None
        self.before_img_pil = None
        
        # --- 3-COLUMN LAYOUT ---
        # 1. Control Panel (Left)
        self.control_canvas = tk.Canvas(root, width=450, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.control_canvas.yview)
        self.control_frame = ttk.Frame(self.control_canvas, padding=10)
        
        self.control_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        )
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.control_canvas.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y, pady=5)
        
        # 2. After Preview Pane (Middle - Next to controls)
        self.after_container = ttk.LabelFrame(root, text="AFTER (Enhanced Preview)")
        self.after_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.after_panel = ttk.Label(self.after_container, anchor="center")
        self.after_panel.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # 3. Before Preview Pane (Right)
        self.before_container = ttk.LabelFrame(root, text="BEFORE (Original Crop)")
        self.before_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.before_panel = ttk.Label(self.before_container, anchor="center")
        self.before_panel.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # --- VARIABLES ---
        self.contrast_var = tk.DoubleVar(value=1.6)
        self.sharpness_var = tk.DoubleVar(value=1.8)
        self.brightness_var = tk.DoubleVar(value=1.0)
        
        self.auto_contrast_var = tk.BooleanVar(value=False)
        self.blue_channel_var = tk.BooleanVar(value=False)
        self.shading_corr_var = tk.BooleanVar(value=False) # Global uneven lighting fix
        
        # Radial Hotspot Filter variables (targeted at top-right glare)
        self.hotspot_enabled = tk.BooleanVar(value=True)
        self.hs_x_var = tk.IntVar(value=78)    # X % position (Top right)
        self.hs_y_var = tk.IntVar(value=15)    # Y % position
        self.hs_radius_var = tk.IntVar(value=35) # Radius %
        self.hs_strength_var = tk.DoubleVar(value=0.4) # Darkening / correction strength
        
        self.crop_left_var = tk.IntVar(value=23)
        self.crop_top_var = tk.IntVar(value=16)
        self.crop_right_var = tk.IntVar(value=84)
        self.crop_bottom_var = tk.IntVar(value=89)
        
        self.build_controls()
        
        # Bind resize events to fill containers dynamically
        self.after_container.bind("<Configure>", self.update_preview)
        self.before_container.bind("<Configure>", self.update_preview)
        self.update_preview()

    def add_header(self, text, reset_command=None):
        frame = ttk.Frame(self.control_frame)
        frame.pack(fill="x", pady=(12, 2))
        lbl = ttk.Label(frame, text=text, font=("Segoe UI", 10, "bold"), foreground="#1f3a93")
        lbl.pack(side=tk.LEFT)
        if reset_command:
            btn = ttk.Button(frame, text="Reset", width=6, command=reset_command)
            btn.pack(side=tk.RIGHT)
        sep = ttk.Separator(self.control_frame, orient="horizontal")
        sep.pack(fill="x", pady=(2, 6))

    def build_controls(self):
        # --- IMAGE ENHANCEMENTS ---
        self.add_header("✨ Image Enhancements", self.reset_enhancements)
        self.create_control("Contrast", self.contrast_var, 0.1, 3.0, 0.1)
        self.create_control("Sharpness", self.sharpness_var, 0.0, 5.0, 0.1)
        self.create_control("Brightness", self.brightness_var, 0.5, 2.0, 0.1)
        
        ttk.Checkbutton(self.control_frame, text="Auto Contrast (Stretch tonal range)", variable=self.auto_contrast_var, command=self.update_preview).pack(anchor="w", pady=2)
        ttk.Checkbutton(self.control_frame, text="Use Blue Channel (Strip yellow paper)", variable=self.blue_channel_var, command=self.update_preview).pack(anchor="w", pady=2)
        ttk.Checkbutton(self.control_frame, text="Global Shading Correction (Flatten creases)", variable=self.shading_corr_var, command=self.update_preview).pack(anchor="w", pady=2)

        # --- RADIAL HOTSPOT FILTER ---
        self.add_header("🎯 Radial Hotspot Filter", self.reset_hotspot)
        ttk.Checkbutton(self.control_frame, text="Enable Radial Hotspot Corrector", variable=self.hotspot_enabled, command=self.update_preview).pack(anchor="w", pady=2)
        self.create_control("Center X (%)", self.hs_x_var, 0, 100, 1)
        self.create_control("Center Y (%)", self.hs_y_var, 0, 100, 1)
        self.create_control("Radius (%)", self.hs_radius_var, 5, 100, 1)
        self.create_control("Correction Strength", self.hs_strength_var, 0.0, 1.0, 0.05)

        # --- CROP BOUNDARIES ---
        self.add_header("📐 Crop Boundaries", self.reset_crops)
        self.create_control("Crop Left %", self.crop_left_var, 0, 50, 1)
        self.create_control("Crop Top %", self.crop_top_var, 0, 50, 1)
        self.create_control("Crop Right %", self.crop_right_var, 50, 100, 1)
        self.create_control("Crop Bottom %", self.crop_bottom_var, 50, 100, 1)
        
        # --- ACTIONS ---
        self.add_header("💾 Output")
        btn = ttk.Button(self.control_frame, text="Print Snippet & Save Preview", command=self.save_snippet)
        btn.pack(fill="x", pady=8)

    def create_control(self, label_text, var, min_val, max_val, resolution):
        frame = ttk.Frame(self.control_frame)
        frame.pack(fill="x", pady=2)
        
        lbl = ttk.Label(frame, text=label_text, width=16, anchor="w", font=("Segoe UI", 9))
        lbl.pack(side=tk.LEFT)
        
        scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, command=lambda _: self.update_preview())
        scale.pack(side=tk.LEFT, expand=True, fill="x", padx=4)
        
        # Spinbox acts as input field + fine control arrows
        spin = ttk.Spinbox(frame, from_=min_val, to=max_val, increment=resolution, textvariable=var, width=6, command=self.update_preview)
        spin.pack(side=tk.RIGHT)
        spin.bind("<Return>", lambda e: self.update_preview())

    def reset_enhancements(self):
        self.contrast_var.set(1.6)
        self.sharpness_var.set(1.8)
        self.brightness_var.set(1.0)
        self.auto_contrast_var.set(False)
        self.blue_channel_var.set(False)
        self.shading_corr_var.set(False)
        self.update_preview()

    def reset_hotspot(self):
        self.hotspot_enabled.set(True)
        self.hs_x_var.set(78)
        self.hs_y_var.set(15)
        self.hs_radius_var.set(35)
        self.hs_strength_var.set(0.4)
        self.update_preview()

    def reset_crops(self):
        self.crop_left_var.set(23)
        self.crop_top_var.set(16)
        self.crop_right_var.set(84)
        self.crop_bottom_var.set(89)
        self.update_preview()

    def apply_pipeline(self, img_pil):
        # 1. Blue channel / Yellow removal
        if self.blue_channel_var.get():
            _, _, blue = img_pil.split()
            img_pil = blue.convert('RGB')
            
        # 2. Global Shading Correction (removes crease shadows and uneven lighting)
        if self.shading_corr_var.get():
            cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
            # Estimate background via large blur
            bg = cv2.GaussianBlur(cv_img, (101, 101), 0)
            # Divide out uneven lighting
            norm = cv2.divide(cv_img, bg, scale=255)
            img_pil = Image.fromarray(norm).convert('RGB')

        # 3. Radial Hotspot Filter (targets specific bright glare like top-right)
        if self.hotspot_enabled.get() and self.hs_strength_var.get() > 0:
            np_img = np.array(img_pil).astype(np.float32)
            h, w, _ = np_img.shape
            cx = int(w * (self.hs_x_var.get() / 100.0))
            cy = int(h * (self.hs_y_var.get() / 100.0))
            radius = int(max(w, h) * (self.hs_radius_var.get() / 100.0))
            
            if radius > 0:
                y_indices, x_indices = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
                
                # Smooth radial mask (1 at center, 0 outside radius)
                mask = np.clip(1.0 - (dist_from_center / radius), 0.0, 1.0)
                mask = mask ** 2 # smoother falloff
                
                strength = self.hs_strength_var.get()
                # Darken/tone down the bright hotspot area
                for c in range(3):
                    np_img[:, :, c] = np_img[:, :, c] * (1.0 - mask * strength * 0.4) + (np_img[:, :, c] * (1.0 - mask * strength)) * 0.5
                np_img = np.clip(np_img, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(np_img)

        # 4. Auto Contrast
        if self.auto_contrast_var.get():
            img_pil = ImageOps.autocontrast(img_pil, cutoff=2)
            
        # 5. Standard Enhancements
        c = self.contrast_var.get()
        if c > 0:
            img_pil = ImageEnhance.Contrast(img_pil).enhance(c)
            
        s = self.sharpness_var.get()
        if s > 0:
            img_pil = ImageEnhance.Sharpness(img_pil).enhance(s)
            
        br = self.brightness_var.get()
        if br > 0:
            img_pil = ImageEnhance.Brightness(img_pil).enhance(br)
            
        return img_pil

    def update_preview(self, event=None):
        width, height = self.original_img.size
        
        l = int(width * (self.crop_left_var.get() / 100.0))
        t = int(height * (self.crop_top_var.get() / 100.0))
        r = int(width * (self.crop_right_var.get() / 100.0))
        b = int(height * (self.crop_bottom_var.get() / 100.0))
        
        if r <= l or b <= t:
            return
            
        # Before Image
        self.before_img_pil = self.original_img.crop((l, t, r, b))
        
        # After Image
        self.after_img_pil = self.apply_pipeline(self.before_img_pil.copy())
        
        # Resize dynamically to fill container dimensions
        self.render_image_to_label(self.before_img_pil, self.before_panel, self.before_container)
        self.render_image_to_label(self.after_img_pil, self.after_panel, self.after_container)

    def render_image_to_label(self, img_pil, label_widget, container_widget):
        c_width = container_widget.winfo_width() - 30
        c_height = container_widget.winfo_height() - 50
        if c_width < 100: c_width = 500
        if c_height < 100: c_height = 700
        
        img_copy = img_pil.copy()
        img_copy.thumbnail((c_width, c_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img_copy)
        label_widget.configure(image=photo)
        label_widget.image = photo

    def save_snippet(self):
        print("\n" + "✨"*5 + " CAPTURED SETTINGS SNIPPET " + "✨"*5)
        print(f"blue_channel = {self.blue_channel_var.get()}")
        print(f"shading_correction = {self.shading_corr_var.get()}")
        print(f"hotspot_enabled = {self.hotspot_enabled.get()}")
        print(f"hotspot_x_pct = {self.hs_x_var.get()}")
        print(f"hotspot_y_pct = {self.hs_y_var.get()}")
        print(f"hotspot_radius_pct = {self.hs_radius_var.get()}")
        print(f"hotspot_strength = {self.hs_strength_var.get():.2f}")
        print(f"contrast = {self.contrast_var.get():.2f}")
        print(f"sharpness = {self.sharpness_var.get():.2f}")
        print(f"brightness = {self.brightness_var.get():.2f}")
        print(f"left = int(width * {self.crop_left_var.get() / 100.0:.2f})")
        print(f"top = int(height * {self.crop_top_var.get() / 100.0:.2f})")
        print(f"right = int(width * {self.crop_right_var.get() / 100.0:.2f})")
        print(f"bottom = int(height * {self.crop_bottom_var.get() / 100.0:.2f})")
        print("✨"*36 + "\n")
        
        if self.after_img_pil:
            self.after_img_pil.save("tuned_preview.jpg", "JPEG", quality=95)
            print("📁 Successfully saved high-res sample to 'tuned_preview.jpg'!\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiaryTunerApp(root, original_img)
    root.mainloop()