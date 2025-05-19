import torch
from torch.utils.data import Dataset
import numpy as np

class CSVDataset_cycle(Dataset):
    """
    A PyTorch Dataset for loading gene expression data from NPY files for CycleGAN training.
    
    This dataset loads gene expression data, perturbed gene indicators, and batch labels
    from a pre-processed NPY file. It uses memory mapping to efficiently handle large datasets
    without loading the entire file into memory.
    """
    
    def __init__(self, npy_file_path):
        """
        Initialize the CSVDataset_cycle.

        Args:
            npy_file_path (str): Path to the merged .npy file for source or target domain.
                                 The file should contain gene expression data, perturbed gene indicators,
                                 and batch labels concatenated along the feature dimension.
        """
        # Load data using memory mapping to save memory
        print(f"Loading data from {npy_file_path}")
        # Shape: [num_samples, gene_size + perturbed_size + batch_size]
        self.data = np.load(npy_file_path, mmap_mode='r')  
        # Determine device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Commented out to avoid loading entire dataset to GPU memory
        # self.data = torch.tensor(self.data, dtype=torch.float32).to(self.device)
        print(f"Dataset loaded on {self.device}")

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: The total number of samples.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            torch.Tensor: A tensor containing gene expression, perturbed gene indicators,
                         and batch labels for the specified sample, moved to the appropriate device.
        """
        return torch.from_numpy(self.data[idx]).float().to(self.device)