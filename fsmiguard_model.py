import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np

class FSMIGuardTransformer(nn.Module):
    def __init__(self, vocab_size=2500, d_model=512, n_layers=6, n_heads=8, max_seq_length=512):
        super(FSMIGuardTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        return self.fc_out(x.mean(dim=1)) # Pooling over sequence length

class GMMAnomalyDetector:
    def __init__(self, n_components=4, threshold_multiplier=2.5):
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.threshold_multiplier = threshold_multiplier
        self.threshold = None

    def fit(self, latent_features):
        self.gmm.fit(latent_features)
        # Compute log-probabilities of the training data to establish baseline threshold
        scores = self.gmm.score_samples(latent_features)
        self.threshold = np.mean(scores) - (self.threshold_multiplier * np.std(scores))

    def predict(self, latent_features):
        scores = self.gmm.score_samples(latent_features)
        # Return 1 for anomaly, 0 for benign
        return (scores < self.threshold).astype(int)
