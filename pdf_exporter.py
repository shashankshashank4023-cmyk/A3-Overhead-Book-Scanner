from PIL import Image
import tempfile, os

def export_pdf(pages, output_path):
    if not pages: raise ValueError("No pages to export.")
    from PIL import Image as PILImage
    converted = []
    for page in pages:
        # OpenCV BGR -> RGB
        import cv2
        rgb = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)
        converted.append(PILImage.fromarray(rgb).convert("RGB"))
    first, rest = converted[0], converted[1:]
    first.save(output_path, "PDF", resolution=300.0, save_all=True, append_images=rest)
