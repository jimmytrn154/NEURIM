"""Real-time EEG-to-digit (0-9) classifier for the EMOTIV EPOC X.

Faithful re-implementation of the BrainDigiCNN 1D-CNN from Tiwari et al. (2023),
trained on the public MindBigData "EPOC" digit dataset and deployed for
windowed real-time inference over the EMOTIV Cortex API.
"""
