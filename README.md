This project implements EEG-based brain intent decoding using machine learning and deep learning.
The goal is to classify motor imagery EEG signals (left-hand vs right-hand intentions) using publicly
available datasets (PhysioNet EEG Motor Imagery).

Pipeline:
1. Data loading and preprocessing
2. Classical baseline using CSP + LDA
3. Deep learning model (EEGNet-style CNN)
4. Evaluation and latency analysis
5. Simulation for control signal mapping
