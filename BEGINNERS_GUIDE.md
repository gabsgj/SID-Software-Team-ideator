# 🐣 The Absolute Beginner's Guide to SID

Welcome! If you are completely new to coding, artificial intelligence, or drone inspection software, this is the perfect place to start. We will walk you through exactly what this software does and how to use it step-by-step.

---

## 🤔 1. What does this software exactly do?

Imagine you have a drone (let's call it **SID** – Structural Inspection Drone). This drone flies up to a large concrete bridge, takes a video, and tries to find cracks on the bridge. 

However, just finding a crack isn't enough for engineers. They need to know exactly how **wide** the crack is (in millimeters) to decide if the bridge is safe or hiding critical damage. 

This software does three main things:
1. **Detects the Crack:** It uses an AI brain (YOLOv11) to draw a tight colored boundary over any pixel that looks like a crack.
2. **Finds the Width:** It draws a single line down the center of the crack (the skeleton) and casts out "measuring tapes" sideways at a 90-degree angle to find the exact local width in pixels.
3. **Converts to Real World Size (mm):** It uses math and a reference object (like a printed QR-like tag) to convert that pixel width into real-world millimeters.

---

## 🛠️ 2. How to get the software ready to use

Even though it sounds complex, using it is just typing a few lines in a chat window called the **Terminal**.

### Step 2.1: Open your Terminal
*   **Mac Users:** Press `Command + Spacebar`, type `Terminal`, and press Enter.
*   **Windows Users:** Press the `Windows Key`, type `cmd`, and press Enter.

### Step 2.2: Make sure you are in the project folder
Type this command and press Enter:
```bash
cd /Users/gabriel/Projects/SID-Software-Team-Ideator
```
*(This just opens the folder in the terminal).*

### Step 2.3: Turn on your "Virtual Environment"
Think of a virtual environment like a sandbox. We want to install the software only inside this sandbox so we don't break anything else on your computer.

Copy and paste this into the terminal and press Enter:
*   **Mac/Linux:** `source venv/bin/activate`
*   **Windows:** `venv\Scripts\activate`

*(You should see `(venv)` appear on the left side of your terminal screen. The sandbox is now ON!)*

---

## 🕹️ 3. Playing with the Quick Demo (No Camera Needed)

Let's test the math engine to make sure everything works. We have a built-in demo that draws fake "synthetic" cracks and measures them. 

Type this into your terminal:
```bash
python3 run_demo.py --mode demo
```

**What just happened?**
The software generated some test images, measured the fake cracks, and printed a table of results. It also created an HTML webpage report! You can find the report in the `inspection_output/` folder.

---

## 📷 4. Testing with your Computer Webcam!

Now for the fun part. You can test it live using your own computer webcam!

Make sure your sandbox `(venv)` is still active, and type:
```bash
python3 run_webcam.py --model best.pt --confidence 0.40
```

*Note for Mac Users: If it fails, your Mac might be blocking the camera. Go to Apple Logo (top left) ➔ System Settings ➔ Privacy & Security ➔ Camera ➔ and turn on access for your Terminal!*

### Webcam Keyboard Controls
If the camera turns on and you see yourself:
*   Try moving close to a crack on a wall (or draw a squiggly black line on a white piece of paper and hold it up to the camera).
*   **Press `+` or `-`** on your keyboard to make the AI more or less sensitive. 
*   **Press `s`** to save a screenshot of what you see.
*   **Press `q`** to close the camera window and quit.

---

## 🔲 5. What is this "ArUco Marker" Calibration?

When you use the webcam, the software tells you the crack is something like `"12.4px"` wide. **px** stands for pixels. But pixels represent the screen, not reality. If you move the camera closer, the pixel number will get bigger, even though the crack didn't grow!

To fix this, we use an **ArUco Marker**.
1. Search online for **"ArUco DICT_4X4_50 ID 0"**.
2. Display it on your phone screen or print it on a piece of paper. (Measure with a real ruler how wide the black box is, e.g., 50 millimeters).
3. Open the webcam app (`python3 run_webcam.py`).
4. Hold the marker next to the crack. 
5. **Magic!** The software will immediately recognize the square, draw a green box around it, and instantly change its measurements from `Pixels` to real-world `Millimeters`!

That's it! You are now fully trained on how to use the SID software suite. 🎉
