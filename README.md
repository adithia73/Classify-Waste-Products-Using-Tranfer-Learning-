
# 🧠 Binary Waste Classification (O vs R) Using VGG16 + Fine-Tuning

This project implements a **binary image classification** model using **Transfer Learning** with the **VGG16** architecture. The model is trained to differentiate between two categories of waste images labeled **'O'** and **'R'**.

---

## 📂 Dataset Structure

The dataset is structured into separate `train` and `test` folders with class subdirectories:

```
o-vs-r-split/
├── train/
│   ├── O/
│   └── R/
└── test/
    ├── O/
    └── R/
```

Each folder contains the corresponding images for that class.

---

## 🔍 Problem Statement

> Develop a binary classifier that can distinguish between images of two different types of waste — 'O' and 'R' — using pre-trained deep learning models for efficient and accurate classification.

---

## 🧪 Approach & Methodology

### 🔄 1. **Transfer Learning with VGG16**

- Base model: **VGG16**, pre-trained on ImageNet.
- `include_top=False` (removes fully connected layers).
- Input shape: `(150, 150, 3)`.
- Custom classification head added on top.

### 🔧 2. **Fine-Tuning**

- Initially freeze base model layers.
- Later unfreeze top layers to fine-tune with a low learning rate.

### 🔢 3. **Data Augmentation**

- Applied random flips, rotations, zoom, and rescaling to prevent overfitting.

### 📊 4. **Evaluation Metrics**

- Accuracy
- Loss curves
- Confusion matrix
- Sample predictions visualization

---

## 🧠 Model Architecture Details

```python
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

input_shape = (150, 150, 3)
vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

# Extract features from VGG16 base
output = vgg.layers[-1].output
output = Flatten()(output)
basemodel = Model(vgg.input, output)

# Custom classification head
model = Sequential()
model.add(basemodel)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
```

---

## ✅ Results

| Phase           | Accuracy |
|-----------------|----------|
| Transfer Phase  | ~95%     |
| Fine-Tuned Phase| ~97%     |

The model achieves high accuracy distinguishing between the two waste classes.

---

## 🛠 Tech Stack

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- scikit-learn  

### Installation

```bash
pip install -r requirements.txt
```

---

## 📁 Project Files

```
.
├── classify-waste-products-using-tl-ft.ipynb   # Jupyter Notebook with full training pipeline
├── o-vs-r-split/                               # Dataset (train/test split)
│   ├── train/
│   └── test/
└── README.md
```

---

## 📌 Key Takeaways

- VGG16, with fine-tuning, is effective for binary waste classification.
- Data augmentation reduces overfitting.
- Transfer learning saves training time and improves performance.

---

## 🚀 Future Improvements

- Experiment with lightweight architectures (e.g., MobileNet).
- Extend to multi-class classification.
- Deploy the model as a web or mobile app.

---

## 👤 Author

**Adithia V**  
AI & Deep Learning Enthusiast  

---
