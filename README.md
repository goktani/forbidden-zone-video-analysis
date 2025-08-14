```markdown
# Forbidden Zone Video Analysis

## 📌 Summary
YOLOv8-based system for detecting and tracking people entering a predefined forbidden zone in videos, with segmentation overlay and warning alerts.

---

## 📖 Description
This project processes `.mp4` videos to detect people and determine if they enter a **forbidden zone**.  
It uses **YOLOv8n-seg** for human segmentation and supports **real-time processing** with GPU acceleration on Mac (M1/M2) via **Metal Performance Shaders (MPS)**.

The system has **three versions**:

1. **`segment_forbidden_overlay.py`**  
   - Applies segmentation to detect people.  
   - Draws a forbidden zone overlay.  
   - Highlights when a detected person overlaps with the zone.

2. **`segment_forbidden_overlay_tracking.py`**  
   - Adds **multi-object tracking** to keep consistent IDs for detected people.  
   - Saves processed videos in the `processed_videos/` folder.

3. **`segment_forbidden_overlay_trackingv2.py`** *(Final Version)*  
   - Allows the **forbidden zone to be selected before each video**.  
   - Optimized for better performance and usability.  
   - Saves processed videos in the `processed_videosv2/` folder.

---

## 📂 Folder Structure
```
````
├── processed\_videos/                  # Outputs from version 2
├── processed\_videosv2/                 # Outputs from final version
├── test\_videos/                        # Input .mp4 videos
│   ├── media1.mp4
│   ├── media2.mp4
│   └── ...
├── segment\_forbidden\_overlay.py        # Version 1 - Segmentation + Overlay
├── segment\_forbidden\_overlay\_tracking.py   # Version 2 - Tracking + Overlay
├── segment\_forbidden\_overlay\_trackingv2.py # Version 3 - Final optimized version
├── yolov8n-seg.pt                       # YOLOv8 segmentation model

````

---

## ⚙️ Requirements
- Python 3.9+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy
- PyTorch (with MPS support on Mac if available)

Install dependencies:
```bash
pip install ultralytics opencv-python numpy torch
````

---

## 🚀 How to Run

### **1️⃣ Version 1 - Basic Segmentation**

```bash
python segment_forbidden_overlay.py
```

* Processes all `.mp4` files in `test_videos/`.
* Detects people, applies segmentation, overlays forbidden zone.

---

### **2️⃣ Version 2 - Tracking Enabled**

```bash
python segment_forbidden_overlay_tracking.py
```

* Same as Version 1 but keeps **consistent IDs** for each person.
* Outputs saved in **`processed_videos/`**.

---

### **3️⃣ Version 3 - Final Version (Selectable Zone)**

```bash
python segment_forbidden_overlay_trackingv2.py
```

* Before each video, lets you **select the forbidden zone manually** with the mouse.
* Outputs saved in **`processed_videosv2/`**.
* Press **`Q`** to stop processing early.

---

## 🖥️ GPU Acceleration on Mac M1/M2

To enable Metal Performance Shaders (MPS) for faster processing:

```python
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
```

This will use Apple’s GPU for significant speed improvement over CPU.

---

## 📌 Notes

* Forbidden zone is **not** detected automatically — you must define it manually in Version 3.
* Segmentation is performed **only on people** for performance reasons.
* Overlap between the forbidden zone and a person triggers a warning overlay.

---

## 📄 License

MIT License - You are free to use, modify, and distribute this project with attribution.

---

## ✨ Author

Developed by **\Göktan İren** as part of a real-time human behavior and safety analysis system.

```

---

