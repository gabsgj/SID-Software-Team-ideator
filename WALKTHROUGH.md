# 🚶‍♂️ SID Step-by-Step Walkthrough Guide
### Easy Setup, Testing, and Live Webcam Demonstration

This guide will walk you through setting up the project and testing the crack detection module from scratch. No advanced computer vision knowledge is required!

---

## 🛠️ Step 1: Setting Up Your Computer

To run this software, you need **Python** installed. We use a **Virtual Environment (venv)** to install packages cleanly without messing up other software on your computer.

Open your **Terminal** (macOS/Linux) or **Command Prompt** (Windows) and type the following commands line by line:

### 1. Go to the project folder
Use the `cd` command to navigate to where this project is located:
```bash
cd /Users/gabriel/Projects/SID-Software-Team-Ideator
```

### 2. Set up the virtual environment
Create a isolated Python sandbox called `venv`:
```bash
python3 -m venv venv
```

### 3. Activate the environment
This tells your terminal to use this project's sandbox:
*   **macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```
*   **Windows:**
    ```cmd
    venv\Scripts\activate
    ```
*(You will see `(venv)` appear at the beginning of your command line, indicating the sandbox is active.)*

### 4. Install the required tools
Installs packages like OpenCV, Ultralytics, PyTest, and others:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏎️ Step 2: Run the Synthetic Demo (Easiest Test)

Before hooking up any camera, you can test if the math engine is working correctly. We have a built-in demo that generates synthetic crack drawings of known widths (e.g., 2, 5, 10 pixels wide) and measures them.

Make sure your virtual environment is active, then run:
```bash
python run_demo.py --mode demo
```

### What happens when you run this:
1. It creates mock cracks.
2. It runs the monocular width estimator to find their widths.
3. It prints a table showing the **True Width** vs the **Measured Width** and computes the error.
4. It compiles everything into an **interactive HTML report**!
5. Open `inspection_output/demo/demo_report.html` in your web browser to view the report dashboard.

---

## 📷 Step 3: Run the Live Webcam Test

You can test the system using your own webcam. It uses the `best.pt` YOLO model file located in the project folder to detect cracks, skeletons them in blue, measures their width, and alerts you to their severity in real time.

With your environment active, run:
```bash
python run_webcam.py --model best.pt --confidence 0.40
```

### ⚠️ Mac Users Troubleshooting (Camera Permission)
If the script starts but exits saying it cannot open the camera, your Terminal does not have camera access:
1. Open **System Settings** on your Mac.
2. Go to **Privacy & Security** ➔ **Camera**.
3. Toggle the switch next to **Terminal** (or VS Code) to **ON**.
4. Restart your terminal and run the command again.

### 🎮 Webcam Controls (Keyboard Shortcuts)
When the webcam window is active, use these keyboard keys to control the application:
*   **`q`** : **Quit** the webcam screen.
*   **`+`** or **`=`** : **Increase Detection Confidence** (filters out false alarms).
*   **`-`** or **`_`** : **Decrease Detection Confidence** (detects fainter, thinner cracks).
*   **`s`** : **Save Screenshot** (saves the annotated camera frame to `inspection_output/screenshots/`).
*   **`c`** : **Clear Scale Calibration** (reverts measurements back to pixels).

---

## 📏 Step 4: Calibrating Pixels to Millimeters (ArUco Marker)

By default, the webcam doesn't know how far it is from the surface, so it measures cracks in **pixels**. 
To measure cracks in **real-world millimeters**, you can calibrate it instantly:

1. **Print an ArUco Marker:**
   *   An ArUco marker is a black-and-white grid pattern that looks like a simplified QR code.
   *   Print an **ArUco marker** from the `DICT_4X4_50` dictionary (e.g., ID 0, search "ArUco DICT_4X4_50 ID 0" online to download/print one, or display it on a smartphone screen).
   *   Measure the physical width of the printed marker with a ruler in millimeters (e.g., exactly 50mm).
2. **Calibrate:**
   *   Hold the printed marker next to a crack in front of your webcam.
   *   The software will automatically detect it, outline it in green, and use its known size to calculate the camera scale dynamically.
   *   All crack width labels on the screen will immediately switch from **pixels (px)** to **millimeters (mm)**!
   *   If you move the camera closer or further away, the millimeter measurements will adapt dynamically as long as the marker is visible.

---

## 🖼️ Step 5: Test on Your Own Saved Images

If you have photos of concrete surfaces with cracks on your computer:

1. Place the photo (e.g., `my_bridge.jpg`) in the project folder.
2. If you know the drone or camera altitude (distance to surface), you can compute millimeters. Run:
   ```bash
   python run_demo.py --mode image --image my_bridge.jpg --altitude 1.5
   ```
   *(Assuming camera is 1.5 meters from the concrete surface)*
3. Check the output folders:
   *   Annotated image with width overlays: `inspection_output/my_bridge/my_bridge_annotated.jpg`
   *   Full HTML dashboard: `inspection_output/my_bridge/my_bridge_report.html`

---

## 🛠️ Step 6: Verify Everything with Automated Tests

To double check that no files are corrupted and all python code compiles and operates flawlessly, run the automated test suite:
```bash
pytest tests/ -v
```
You should see all **55 passed** tests in green!

---

*IDEATOR GECT · SID Software Engineering Team*
