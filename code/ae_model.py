"""Autoencoder architecture and helpers shared by the training script (autoe.py) and the projection script."""
import numpy as np, pandas as pd, torch, torch.nn as nn

# HGNC symbol aliases between the TCGA (PanCanAtlas) and CPTAC (GDC STAR) gene annotations
ALIAS_MAP = {
    'ACKR1': 'DARC', 'ADGRE1': 'EMR1', 'ADGRF5': 'GPR116', 'ADGRV1': 'GPR98', 'CCN3': 'NOV', 'CCN5': 'WISP2', 'CCN6': 'WISP3', 'CCNP': 'CNTD2',
    'CES4A': 'CES7', 'CLTRN': 'CLTB', 'CYP2D7': 'CYP2D7P1', 'DRC3': 'LRRC48', 'ERICH5': 'C14orf37', 'FAM30A': 'KIAA0125', 'FAXDC2': 'C5orf4', 'FFAR4': 'GPR120',
    'GALNT15': 'GALNTL2', 'GUCY1A1': 'GUCY1A3', 'KIRREL1': 'KIRREL', 'P3H2': 'LEPREL1', 'PIANP': 'LRRTM4', 'PLPPR1': 'LPPR1', 'PLPPR5': 'LPPR5', 'SYNDIG1': 'TMEM90B',
    'TENM1': 'ODZ1', 'TENM4': 'ODZ4', 'CCER2': 'C10orf53', 'SAXO1': 'FAM154A',
}


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.3):
        super().__init__()
        enc, prev = [], input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]; prev = h
        enc.append(nn.Linear(prev, latent_dim)); self.encoder = nn.Sequential(*enc)
        dec, prev = [], latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]; prev = h
        dec.append(nn.Linear(prev, input_dim)); self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x); return self.decoder(z), z


def load_bundle(path):
    b = torch.load(path, map_location="cpu", weights_only=False); cfg = b["config"]
    m = Autoencoder(cfg["input_dim"], cfg["hidden_dims"], cfg["latent_dim"], cfg.get("dropout", 0.3)); m.load_state_dict(b["state_dict"]); m.eval()
    return b, m


def encode_to_pcs(bundle, model, X_log2_by_gene):
    """X_log2_by_gene: (n_samples, 3000) array in the bundle's gene order, log2(expr+1). Returns the PC1-PC10 scores."""
    sm = np.asarray(bundle["scaler_mean"], np.float32); ss = np.asarray(bundle["scaler_scale"], np.float32)
    Z = (np.asarray(X_log2_by_gene, np.float32) - sm) / (ss + 1e-8)
    with torch.no_grad(): z = model.encoder(torch.FloatTensor(Z)).numpy()
    P = (z - np.asarray(bundle["pca10_mean"])) @ np.asarray(bundle["pca10_components"]).T
    return pd.DataFrame(P, columns=[f"AE_PC{i + 1}" for i in range(10)]), z
