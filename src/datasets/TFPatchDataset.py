import numpy as np
from sklearn.preprocessing import StandardScaler

from src.datasets.Dataset import Dataset

class TFPatchDataset(Dataset):
:nt
:nt
q
visual
q
visual
:visual
§:wq
    """ 
      A data loader for time-frequency patches.
      features2d: list of 2D TF features of (n_frames, n_freq_bins)
      labels: list of labels where indices correspond to features2d
      patch_len: length of each time frequency input patch
      patch_hop: hop length between each patch
      floatx: float type
    """
    def __init__(self, file_features, file_labels, patch_len, patch_hop, floatx=np.float32, scaler=None):
        self.file_features   = file_features
        self.file_labels     = file_labels
        self.patch_len  = patch_len
        self.patch_hop  = patch_hop
        self.n_freq_bins = files_features[0][-1]

        self.floatx = floatx
        self.scaler = scaler # reuse training scaler for validation generator

        self.get_patches()

        # normalize the data across mel bins for all time points
        self.features2d = self.features.reshape(-1, self.features.shape[2])
        if scaler is None:
            self.scaler = StandardScaler()
            self.features2d = self.scaler.fit_transform(self.features2d)
        else:
            self.features2d = self.transform(self.features2d)

    def get_patches(self):
        self.num_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_clip(clip)
                    for clip in self.file_features], dtype=int)
        )

        self.num_inst_total = self.num_inst_cum[-1] # number of TF patches

        self.features = np.zeros((self.num_inst_total, self.patch_len, self.n_freq_bins), dtype=self.floatx)
        self.labels   = np.zeros((self.num_inst_total, 1), dtype=self.floatx)

        for idx in range(len(self.file_features)):
            self.features2d_to_patches(idx)

    def features2d_to_patches(self, idx):
        """ Convert a time frequency clip to input patches
        """
        idx_start = self.num_inst_cum[idx]
        idx_end   = self.num_inst_cum[idx + 1]

        idx = 0
        start = 0
        while idx < (idx_end - idx_start):
            self.features[idx_start + idx] = self.file_features[idx][start: start + self.patch_len]

            start += self.patch_hop
            idx += 1
        self.labels[idx_start:idx_end] = self.file_labels[idx]

    def get_num_instances_per_clip(self, clip):
        """ Get the number of time frequency patches in a given clip
        """
        n_frames = clip.shape[0]
        return np.maximum(1, int(np.ceil((n_frames - self.patch_len) / self.patch_hop)))

    def __getitem__(self, index):
        sample = {
            "features": np.expand_dims(self.features[index], -1),
            "labels": self.labels[idx]
        }
        return sample

    def __len__(self):  
        return self.num_inst_total
