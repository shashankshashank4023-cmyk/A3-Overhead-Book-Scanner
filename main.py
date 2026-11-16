import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import json
import threading
from camera import Camera
from processor import process_page, split_spread
from pdf_exporter import export_pdf

APP_TITLE = "A3 Overhead Book Scanner"

class ScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x820")
        self.camera = Camera()
        self.frame = None
        self.preview = None
        self.pages = []
        self.current_image = None
        self.undo_stack = []
        self.session_dir = Path("sessions") / "session"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.mode = tk.StringVar(value="Document")
        self.dual = tk.BooleanVar(value=False)
        self.source = tk.StringVar(value="Camera")
        self._build_ui()
        self.root.bind("<space>", lambda e: self.capture())
        self.root.bind("z", lambda e: self.undo())
        self.root.bind("s", lambda e: self.save_session())
        self.root.bind("m", lambda e: self.next_mode())
        self.root.bind("p", lambda e: self.make_pdf())
        self.root.bind("f", lambda e: self.open_folder())
        self.root.bind("a", lambda e: self.toggle_auto())
        self.root.bind("<Escape>", lambda e: self.stop_camera())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.refresh)

    def _build_ui(self):
        bar = ttk.Frame(self.root, padding=8); bar.pack(fill="x")
        ttk.Button(bar, text="Start Camera", command=self.start_camera).pack(side="left", padx=3)
        ttk.Button(bar, text="Stop Camera", command=self.stop_camera).pack(side="left", padx=3)
        ttk.Button(bar, text="Open Image Folder", command=self.open_folder).pack(side="left", padx=3)
        ttk.Button(bar, text="Capture  [Space]", command=self.capture).pack(side="left", padx=3)
        ttk.Button(bar, text="Undo  [Z]", command=self.undo).pack(side="left", padx=3)
        ttk.Button(bar, text="Save Session  [S]", command=self.save_session).pack(side="left", padx=3)
        ttk.Button(bar, text="Export PDF  [P]", command=self.make_pdf).pack(side="left", padx=3)
        opts = ttk.Frame(self.root, padding=(8,0)); opts.pack(fill="x")
        ttk.Label(opts, text="Enhancement:").pack(side="left")
        ttk.Combobox(opts, textvariable=self.mode, values=["Document","Color","Grayscale"], width=13, state="readonly").pack(side="left", padx=5)
        ttk.Checkbutton(opts, text="Split dual-page spread", variable=self.dual).pack(side="left", padx=12)
        ttk.Label(opts, text="Auto detect: press A").pack(side="left", padx=15)
        self.status = tk.StringVar(value="Ready. Connect a UVC camera or open a folder of sample images.")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w").pack(side="bottom", fill="x")
        body = ttk.Panedwindow(self.root, orient="horizontal"); body.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(body); right = ttk.Frame(body); body.add(left, weight=4); body.add(right, weight=1)
        self.canvas = tk.Label(left, bg="#202020"); self.canvas.pack(fill="both", expand=True)
        ttk.Label(right, text="Captured pages", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(right, width=28); self.listbox.pack(fill="both", expand=True, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.select_page)
        ttk.Button(right, text="Remove selected", command=self.remove_page).pack(fill="x")

    def start_camera(self):
        try:
            self.camera.start()
            self.source.set("Camera")
            self.status.set("Camera started. Space captures a frame; camera requests 3840×2160 MJPG where supported.")
        except Exception as e:
            messagebox.showerror("Camera error", str(e))

    def stop_camera(self):
        self.camera.stop()
        self.status.set("Camera stopped.")

    def open_folder(self):
        folder = filedialog.askdirectory(title="Choose image folder")
        if not folder: return
        paths = sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tif",".tiff")])
        if not paths:
            messagebox.showinfo("No images", "No supported images were found in that folder."); return
        self.mock_images = paths
        self.mock_index = 0
        self.source.set("Folder")
        self.load_mock()
        self.status.set(f"Mock image-folder mode: {len(paths)} image(s). Use Space to process current image.")

    def load_mock(self):
        if not getattr(self, "mock_images", []): return
        p = self.mock_images[self.mock_index % len(self.mock_images)]
        self.frame = cv2.imread(str(p))
        self.status.set(f"Folder image {self.mock_index+1}/{len(self.mock_images)}: {p.name}")

    def refresh(self):
        if self.source.get() == "Camera" and self.camera.running:
            frame = self.camera.read()
            if frame is not None: self.frame = frame
        if self.frame is not None:
            self.show_frame(self.frame)
        self.root.after(100, self.refresh)

    def show_frame(self, frame):
        h,w = frame.shape[:2]
        scale = min(1000/w, 680/h)
        display = cv2.resize(frame, (max(1,int(w*scale)), max(1,int(h*scale))))
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.preview = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=self.preview)

    def capture(self):
        if self.frame is None:
            messagebox.showwarning("No source", "Start the camera or open an image folder first."); return
        raw = self.frame.copy()
        try:
            results = process_page(raw, self.mode.get())
            if self.dual.get():
                output_pages = split_spread(results)
            else: output_pages = [results]
            for img in output_pages:
                self.undo_stack.append(list(self.pages))
                self.pages.append(img)
            self._refresh_list()
            self.current_image = self.pages[-1]
            self.show_frame(self.current_image)
            self.status.set(f"Captured {len(output_pages)} page(s). Total: {len(self.pages)}")
            if self.source.get() == "Folder":
                self.mock_index = (self.mock_index + 1) % len(self.mock_images)
                self.load_mock()
        except Exception as e:
            messagebox.showerror("Processing error", str(e))

    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i in range(len(self.pages)): self.listbox.insert(tk.END, f"Page {i+1}")

    def select_page(self, _=None):
        sel = self.listbox.curselection()
        if sel:
            self.current_image = self.pages[sel[0]]
            self.show_frame(self.current_image)

    def undo(self):
        if not self.undo_stack:
            self.status.set("Nothing to undo."); return
        self.pages = self.undo_stack.pop()
        self._refresh_list()
        self.status.set("Last capture undone.")
        if self.pages: self.current_image = self.pages[-1]; self.show_frame(self.current_image)

    def remove_page(self):
        sel = self.listbox.curselection()
        if not sel: return
        self.undo_stack.append(list(self.pages))
        del self.pages[sel[0]]
        self._refresh_list()
        self.status.set("Selected page removed.")

    def save_session(self):
        if not self.pages:
            messagebox.showinfo("Empty session", "Capture at least one page first."); return
        folder = filedialog.askdirectory(title="Choose session output folder")
        if not folder: return
        out = Path(folder) / "scanner_session"; out.mkdir(exist_ok=True)
        entries = []
        for i, img in enumerate(self.pages, 1):
            path = out / f"page_{i:03d}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            entries.append({"page": i, "file": path.name, "width": int(img.shape[1]), "height": int(img.shape[0])})
        (out / "session.json").write_text(json.dumps({"app": APP_TITLE, "pages": entries}, indent=2), encoding="utf-8")
        self.status.set(f"Session saved to {out}")

    def make_pdf(self):
        if not self.pages:
            messagebox.showinfo("Empty session", "Capture pages before exporting."); return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF","*.pdf")])
        if not path: return
        try:
            export_pdf(self.pages, path)
            self.status.set(f"PDF exported: {path}")
            messagebox.showinfo("Export complete", f"PDF saved to:\n{path}")
        except Exception as e: messagebox.showerror("PDF export error", str(e))

    def next_mode(self):
        modes = ["Document","Color","Grayscale"]
        self.mode.set(modes[(modes.index(self.mode.get())+1)%len(modes)])
        self.status.set(f"Enhancement mode: {self.mode.get()}")

    def toggle_auto(self):
        self.status.set("Automatic page detection is enabled during every capture.")

    def close(self):
        self.camera.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScannerApp(root)
    root.mainloop()
