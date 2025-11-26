# 📘 Sequencer2D: Re-Implementing “Sequencer: Deep LSTM for Image Classification”

## ⭐ Overview
This project re-implements and extends parts of the paper *“Sequencer: Deep LSTM for Image Classification”*, replacing Transformer self-attention with **2D BiLSTM sequence modeling** for image classification.

Due to hardware constraints (MacBook Pro + HP, 8GB RAM each), models were trained on **4 classes of CIFAR-10**. Each model took several days to train, and attempts on the full dataset caused slowdowns or kernel crashes.

This repository includes:

- **Sequencer2D Small**
- **Sequencer2D Medium**
- **Vanilla BiLSTM model**
- One **unified notebook** with all models
- Separate notebooks for each model (optional)

---

## 🚀 Model Architectures

### 🔹 1. Sequencer2D Small
A **ViT-like architecture**, replacing self-attention with **BiLSTM2D** (horizontal + vertical directional processing).  

**Pipeline:**
1. Patch Embedding (8×8 → tokens)  
2. LayerNorm  
3. Sequencer Block 1: 4 × Sequencer2D layers  
4. Patch Merging  
5. Sequencer Block 2: 3 × Sequencer2D layers  
6. Pointwise Linear  
7. Adaptive Average Pooling  
8. Fully Connected + Dropout  

**Adjustments:**
- Parameters adapted for CIFAR-10 (32×32)  
- LayerNorm for training stability  
- He initialization  
- Replaced ops that interrupt gradient flow  

---

### 🔹 2. Sequencer2D Medium
Similar to Small but **deeper**, with an extra Sequencer block.  

**Pipeline:**
1. Patch Embedding → LayerNorm  
2. Sequencer Block 1: 4 × Sequencer2D  
3. Patch Merging  
4. Sequencer Block 2: 3 × Sequencer2D  
5. Pointwise Linear  
6. Sequencer Block 3: 3 × Sequencer2D  
7. Pointwise Linear  
8. Adaptive Average Pooling  
9. Fully Connected + Dropout  

**Benefit:** More modeling capacity → slightly better accuracy

---

### 🔹 3. Vanilla BiLSTM Model
Simpler architecture applying **1D BiLSTMs** on flattened image sequences.  

**Pipeline:**
1. Image Flattening  
2. BiLSTM Layers  
3. Fully Connected → Classification Head  

**Performance:** Lower accuracy, confirming the importance of **2D spatial modeling**.

---

## 📊 Results

| Model                  | Avg Loss | Accuracy |
|------------------------|----------|----------|
| Sequencer2D Small      | 0.4246   | 84.11%   |
| Sequencer2D Medium     | 0.4407   | 84.85%   |
| Vanilla BiLSTM         | 0.9368   | 67.77%   |

---

## 🧠 Conclusion
- **Sequencer2D Small & Medium** perform strongly and almost identically.  
- Extra depth (Medium) gives slight improvement under 8GB RAM constraints.  
- **Vanilla BiLSTM** performs worse, showing the need for **2D directional modeling**.  
- With more compute and tuning, Sequencer2D models could surpass these results.

---

## 🔍 Relation to LLM-style Modeling
Sequencer2D shares concepts with **large language models (LLMs):**

- **Sequence Processing:** Images → patch sequences, similar to LLM token streams  
- **Recurrent Modeling vs. Attention:** BiLSTM2D captures 2D spatial relationships  
- **Positional Understanding:** Spatial recurrence encodes position implicitly  
- **Patch Embedding ≈ Tokenization**  
- **Gradient Stability:** LayerNorm, He init, dropout, used in both LSTMs and LLMs  
- **Hybrid Models:** Fits emerging non-attention sequence-processing architectures

---

## 📦 Installation
```bash

pip install -r requirements.txt 

## 📦 Installation, Usage & References

```text
# Installation & Requirements
# Install dependencies:
pip install -r requirements.txt

# requirements.txt should include:
torch==1.13.1
torchvision==0.14.1
scikit-learn==1.1.3
numpy==1.23.5
matplotlib==3.6.3
pandas==1.5.3
torchviz
pickle5

---

# Usage
# Run any notebook directly in the notebooks/ folder:
notebooks/
├── Sequencer2D_Vanilla_all_models.ipynb
├── SequencerModelchanged384M.ipynb
├── SequencerModelchanged384S.ipynb
└── VanillaSequencer_Plain.ipynb

# Each notebook includes:
- Model definition
- Training loop
- Evaluation
- Plots and results

---

# 📚 References
- Main paper: Sequencer: Deep LSTM for Image Classification
- Model parameters adapted from the paper for CIFAR-10 (32×32)
- He initialization and gradient-flow fixes added for stability

