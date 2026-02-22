"""
GUI for pneumonia detection: select an image and get Normal/Pneumonia prediction.
Requires: trained model (resnet18_pneumonia.pth)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

from config import Config
from model import get_model


# Same preprocessing as training
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class PneumoniaPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumonia Detector")
        self.root.geometry("500x550")
        self.root.resizable(True, True)

        self.model = None
        self.current_image_path = None
        self.photo_img = None

        self._load_model()

        self._build_ui()

    def _load_model(self):
        """Load trained model."""
        try:
            self.model = get_model().to(Config.device)
            self.model.load_state_dict(
                torch.load(Config.model_save_path, map_location=Config.device)
            )
            self.model.eval()
        except FileNotFoundError:
            messagebox.showerror(
                "Model Not Found",
                f"Could not find {Config.model_save_path}. Please run train.py first."
            )
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.quit()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(main_frame, text="Pneumonia Detection", font=("Segoe UI", 18, "bold"))
        title.pack(pady=(0, 10))

        subtitle = ttk.Label(main_frame, text="Select a chest X-ray image to classify", font=("Segoe UI", 10))
        subtitle.pack(pady=(0, 20))

        # Select button
        self.btn_select = ttk.Button(main_frame, text="Select Image", command=self._select_image)
        self.btn_select.pack(pady=5, ipadx=20, ipady=8)

        # Image preview
        preview_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=15)

        self.lbl_preview = ttk.Label(preview_frame, text="No image selected", anchor=tk.CENTER)
        self.lbl_preview.pack(fill=tk.BOTH, expand=True, pady=20, padx=20)

        # Analyze button
        self.btn_analyze = ttk.Button(main_frame, text="Analyze", command=self._analyze, state=tk.DISABLED)
        self.btn_analyze.pack(pady=15, ipadx=30, ipady=8)

        # Result area
        result_frame = ttk.LabelFrame(main_frame, text="Result", padding="15")
        result_frame.pack(fill=tk.X, pady=10)

        self.lbl_result = ttk.Label(result_frame, text="—", font=("Segoe UI", 16, "bold"), anchor=tk.CENTER)
        self.lbl_result.pack(fill=tk.X, pady=5)

        self.lbl_confidence = ttk.Label(result_frame, text="", font=("Segoe UI", 11), anchor=tk.CENTER)
        self.lbl_confidence.pack(fill=tk.X)

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        try:
            img = Image.open(path).convert("RGB")
            # Resize for preview (keep aspect ratio, max 200px)
            try:
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            except AttributeError:
                img.thumbnail((200, 200), Image.LANCZOS)
            self.photo_img = self._pil_to_photo(img)
            self.lbl_preview.configure(image=self.photo_img, text="")
            self.lbl_preview.image = self.photo_img
            self.current_image_path = path
            self.btn_analyze.configure(state=tk.NORMAL)
            self.lbl_result.configure(text="—")
            self.lbl_confidence.configure(text="")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def _pil_to_photo(self, pil_image):
        """Convert PIL Image to Tkinter PhotoImage."""
        import io
        from PIL import ImageTk
        return ImageTk.PhotoImage(pil_image)

    def _analyze(self):
        if not self.current_image_path:
            return

        try:
            img = Image.open(self.current_image_path)
            img_tensor = TRANSFORM(img).unsqueeze(0).to(Config.device)

            with torch.no_grad():
                logit = self.model(img_tensor).squeeze()
                prob = torch.sigmoid(logit).item()

            pred = "Pneumonia" if prob >= 0.5 else "Normal"
            confidence = prob if pred == "Pneumonia" else (1 - prob)
            confidence_pct = confidence * 100

            self.lbl_result.configure(text=pred)
            self.lbl_confidence.configure(text=f"Confidence: {confidence_pct:.1f}%")

            # Color feedback
            if pred == "Pneumonia":
                self.lbl_result.configure(foreground="#c0392b")
            else:
                self.lbl_result.configure(foreground="#27ae60")

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")


def main():
    root = tk.Tk()
    app = PneumoniaPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
