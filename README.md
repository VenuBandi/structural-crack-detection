# Structural Crack Detection Using Deep Learning
### Transfer Learning on 56,000+ Infrastructure Images — ResNet50 vs DenseNet121 vs EfficientNet-B0

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-models-orange)
![Dataset](https://img.shields.io/badge/Dataset-SDNET2018-green)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-blueviolet)

> A deep learning pipeline that detects structural cracks in concrete surfaces (bridges, pavements, walls) using transfer learning. Three pretrained CNNs are benchmarked, with the best model further tuned via grid search — prioritizing recall to minimize missed cracks in safety-critical inspection scenarios.

---

## The Problem

Undetected cracks in infrastructure — bridges, roads, buildings — can lead to catastrophic failures. Manual inspection is slow, inconsistent, and expensive. This project builds an automated image classifier that flags cracked concrete surfaces across three surface types using computer vision.

**Missing a crack is worse than a false alarm** — so this project prioritizes **recall** alongside accuracy.

---

## Dataset — SDNET2018

| Surface | Cracked | Uncracked |
|---|---|---|
| Bridge Decks | 2,025 | 11,595 |
| Pavements | 2,608 | 21,726 |
| Walls | 3,851 | 14,287 |
| **Total** | **8,484** | **47,608** |

~56,092 images total. Heavy class imbalance (5.6:1) addressed using **class-weighted cross-entropy loss**.

### Sample Images from Dataset
![Sample Crack Images](reports/images/sample_crack_images.png)
![Sample No-Crack Images](reports/images/sample_nocrack_images.png)

### Class Distribution
![Class Distribution](reports/images/class_distribution.png)

---

## Pipeline

```
SDNET2018 Images (56,092)
        │
        ▼
  ┌───────────────────────────────────────────┐
  │  Preprocessing                            │
  │  • Resize to 224×224                      │
  │  • Augmentation (flip, rotate, jitter)    │
  │  • ImageNet normalization                 │
  │  • 70/15/15 stratified split              │
  └───────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────┐
  │  Transfer Learning — 3 Models             │
  │  • ResNet50 (fine-tuned)                  │
  │  • DenseNet121 (fine-tuned)               │
  │  • EfficientNet-B0 (fine-tuned) ← Best   │
  └───────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────┐
  │  Hyperparameter Tuning (EfficientNet-B0)  │
  │  • Grid search: LR × Batch Size           │
  │  • Best: LR=5e-5, Batch=16               │
  └───────────────────────────────────────────┘
```

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| ResNet50 | 0.9019 | 0.634 | 0.831 | 0.719 |
| DenseNet121 | 0.8965 | 0.613 | 0.856 | 0.714 |
| **EfficientNet-B0** | **0.925** | **0.75+** | **0.83+** | **Best** |
| Tuned EfficientNet-B0 | 0.9166 | 0.680 | **0.848** | 0.754 |

**Winner: EfficientNet-B0** — highest accuracy and F1. Tuned version improves recall (fewer missed cracks) at a slight precision cost.

### Model Comparison — Test Accuracy
![Model Comparison](reports/images/model_comparison_accuracy.png)

### ROC-AUC Comparison (All 4 Models)
![ROC AUC](reports/images/roc_auc_comparison.png)

---

## Training Curves

### ResNet50
![ResNet50 Training](reports/images/resnet50_training_curves.png)

### DenseNet121
![DenseNet121 Training](reports/images/densenet121_training_curves.png)

---

## Confusion Matrices

### ResNet50
![ResNet50 Confusion Matrix](reports/images/resnet50_confusion_matrix.png)

### DenseNet121
![DenseNet121 Confusion Matrix](reports/images/densenet121_confusion_matrix.png)

### EfficientNet-B0
![EfficientNet Confusion Matrix](reports/images/efficientnet_confusion_matrix.png)

---

## Code Highlights

### Custom PyTorch Dataset with Augmentation

```python
from torch.utils.data import Dataset
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class CrackDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.df.loc[idx, "image_path"]).convert("RGB")
        label = self.df.loc[idx, "label"]
        if self.transform:
            image = self.transform(image)
        return image, label
```

### Class-Weighted Loss to Handle Imbalance

```python
# 5.6:1 imbalance — weight the minority (crack) class higher
class_counts = train_df["label"].value_counts().sort_index().values
class_weights = len(train_df) / (2 * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### EfficientNet-B0 Transfer Learning

```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Replace classifier head for binary output
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

### Hyperparameter Grid Search

```python
learning_rates = [1e-4, 5e-5, 1e-5]
batch_sizes = [16, 32]

# Best found: LR=5e-5, Batch=16
# Val Accuracy: 0.9355 | Val F1: 0.7944 | Val Recall: 0.8240
```

---

## Key Findings

- **EfficientNet-B0 is the best model** — highest accuracy (~0.925) and strongest balance of precision/recall
- **DenseNet121 catches the most cracks** (highest raw recall = 0.856) but at the cost of more false alarms
- **Class imbalance is a real challenge** — without weighted loss, models default to predicting "No Crack" and get misleadingly high accuracy
- **Tuning further boosts recall to 0.848** — critical for safety applications where missing a crack matters most

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, torchvision (ResNet50, DenseNet121, EfficientNet-B0) |
| Data Handling | Pandas, PIL, scikit-learn (train/val/test split) |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC |
| Visualization | Matplotlib, Seaborn |
| Dataset | [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) |

---

## Dataset Setup

Download SDNET2018 from [Utah State University Digital Commons](https://digitalcommons.usu.edu/all_datasets/48/) and extract to:

```
~/Desktop/SDNET2018/
├── D/
│   ├── CD/   ← Cracked Decks
│   └── UD/   ← Uncracked Decks
├── P/
│   ├── CP/   ← Cracked Pavements
│   └── UP/
└── W/
    ├── CW/   ← Cracked Walls
    └── UW/
```

## How to Run

```bash
git clone https://github.com/VenuBandi/structural-crack-detection.git
cd structural-crack-detection
pip install -r requirements.txt
jupyter lab Structural_Crack_Detection.ipynb
```

---

## Author

**Venu Bandi** — MS Data Science  
[GitHub](https://github.com/VenuBandi)
