📘 README.md – Sequencer2D: Re-Implementing “Sequencer: Deep LSTM for Image Classification”
⭐ Overview

This project re-implements and extends parts of the paper “Sequencer: Deep LSTM for Image Classification”, replacing Transformer self-attention with 2D BiLSTM sequence modeling for image classification.

The models were trained on 4 classes of the CIFAR-10 dataset due to hardware constraints (MacBook Pro + HP laptop, both 8GB RAM).
Training each model required several days, and attempts at full CIFAR-10 training caused slowdowns or kernel crashes.

The repository includes:

Sequencer2D Small

Sequencer2D Medium

Vanilla BiLSTM model

One unified notebook containing all models

Independent notebooks for each model (optional)

🚀 Model Architectures
🔹 1. Sequencer2D Small

A ViT-like architecture, but replacing self-attention with BiLSTM2D (horizontal + vertical directional sequence processing).

Pipeline:

Patch Embedding (8×8 patches → patch tokens)

LayerNorm

Sequencer Block 1: 4 × Sequencer2D layers

Patch Merging

Sequencer Block 2: 3 × Sequencer2D layers

Pointwise Linear

Adaptive Average Pooling

Fully Connected + Dropout

Adjustments:

Parameters adapted for CIFAR-10 resolution (32×32)

LayerNorm for training stability

He initialization to prevent early stagnation

Removed or replaced ops interrupting gradient flow (e.g., torch.tensor() → .clone())

🔹 2. Sequencer2D Medium

Same core idea as the “Small” model, but deeper, with one additional sequence block.

Pipeline:

Patch Embedding

LayerNorm

Sequencer Block 1: 4 × Sequencer2D

Patch Merging

Sequencer Block 2: 3 × Sequencer2D

Pointwise Linear

Sequencer Block 3: 3 × Sequencer2D

Pointwise Linear

Adaptive Average Pooling

Fully Connected + Dropout

Benefits:

More modeling capacity

Slightly better accuracy

Still constrained by 8GB RAM training environment

🔹 3. Vanilla BiLSTM Model

A simpler architecture that applies 1D BiLSTMs on flattened image sequences (no 2D directional modeling).

Pipeline:

Image Flattening

BiLSTM Layers

Fully Connected

Classification Head

Performance is significantly lower, confirming findings from the original paper.

📊 Results
Model	Average Loss	Accuracy
Sequencer2D Small	0.4246	84.11%
Sequencer2D Medium	0.4407	84.85%
Vanilla BiLSTM	0.9368	67.77%
🧠 Conclusion

Sequencer2D Small and Medium achieve strong and nearly identical performance, validating the paper’s approach.

Additional depth (Medium) improves accuracy slightly but provides diminishing returns under hardware constraints.

The Vanilla BiLSTM performs much worse, confirming:

lack of 2D spatial modeling

inability to capture local-global interactions

poorer gradient flow and spatial representation

With more compute, data, and architectural tuning, Sequencer2D models could likely surpass these results.

🔍 Relation to LLM-style Modeling

Although this is a vision task, Sequencer2D conceptually aligns with large language model (LLM) techniques:

✔ 1. Sequence Processing

LLMs (GPT, LLaMA, etc.) process token sequences.
Sequencer2D transforms images into patch sequences, much like ViTs or LLM token streams.

✔ 2. Recurrent Modeling vs. Attention

Before Transformers, LSTMs were standard in NLP.
Here, BiLSTM2D is applied spatially, making the image act like a 2D “sentence”.

✔ 3. Positional Understanding

LLMs depend on positional encodings.
Sequencer2D implicitly encodes spatial relationships through 2D directional recurrence.

✔ 4. Patch Embedding = Tokenization

Patch embedding for images parallels text tokenization in LLMs.

✔ 5. Gradient Stability Techniques

LayerNorm, He init, and dropout are used in both:

LSTM-based models

Transformer-based LLMs

✔ 6. Hybrid Models

Recent research explores combining:

CNNs

LSTMs

Attention

MLP-Mixers

Vision Transformers

Sequencer2D fits into the emerging category of non-attention sequence-processing architectures, which can be computationally cheaper than Transformers.

📦 Installation
pip install -r requirements.txt


Requirements include:

PyTorch 1.13.1

Torchvision 0.14.1

scikit-learn 1.1.3

NumPy 1.23.5

Matplotlib 3.6.3

Pandas 1.5.3

torchviz

pickle5

▶️ Usage

Run any notebook directly:

notebooks/
├── Sequencer2D_Vanilla_all_models.ipynb
├── SequencerModelchanged384M.ipynb
├── SequencerModelchanged384S.ipynb
└── VanillaSequencer_Plain.ipynb


Each notebook includes:

Model definition

Training loop

Evaluation

Plots

Results

📚 References

Main paper:
Sequencer: Deep LSTM for Image Classification
(All model architecture parameters come from the paper when applicable.)

Note:
This implementation is inspired by the paper’s descriptions.
Parameter counts and block structures were adapted for CIFAR-10 resolution (32×32).
He initialization and several gradient-flow fixes were added to stabilize training.
