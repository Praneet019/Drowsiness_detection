# 🚦 Drowsiness & Age Detection System

This project is a real-time system for detecting driver drowsiness (open/closed eyes) and predicting estimated age from webcam input. It includes two deep learning models and a unified GUI.


## 🛠 Requirements

Install dependencies from requirements.txt:
pip install -r requirements.txt

Make sure you have:
- Python 3.8+
- Webcam enabled (for GUI)
- OS: Windows/Linux/Mac

---

## 📝 Model Training

- Drowsiness Model
  - Notebook: Drowsiness_Detection/drowsiness_model_training.ipynb
  - Dataset: Open and closed eye images (train/test split)
  - Final model saved as drowsiness_model.h5

- Age Model
  - Notebook: Age_Detection/age_model_training.ipynb
  - Dataset: UTKFace aligned dataset, binned into 20-year intervals (0- 20, 21- 40, 41- 60, 61+)
  - Final model saved as final_age_model_4bins.h5

---

## 🚨 How to Run the GUI

After training both models and placing them in the same folder as drowsiness_gui.py, run:
python Drowsiness_Detection/drowsiness_gui.py

What it does:
- Detects faces & eyes
- Predicts if eyes are open or closed (drowsy)
- Predicts age bin (e.g., "21-30")
- Displays real-time video with detection overlays
- Shows a pop-up alert if someone is sleeping

Press q to quit the GUI.

---

## 📊 Metrics

- Drowsiness model accuracy: ~93% on test set
- Age prediction model accuracy: ~77% (10-bin classification)

---

## 📢 Acknowledgements

- UTKFace dataset for age prediction
- Open/closed eyes datasets for drowsiness detection
- TensorFlow & OpenCV for implementation

---

## 📬 Contact

For any issues or feedback, please open an issue or contact me directly.

🚀 Thank you for checking out this project!
