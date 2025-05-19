import os
import glob
import random
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from ..util.utils import get_full_gene_list, get_batch_label_list
import argparse

# -----------------------
# Data Processing Functions
# -----------------------

def process_csv(file_path, full_gene_list, batch_label_list):
    """
    Process a single CSV file to extract gene expression, perturbed genes, and batch labels.
    
    Args:
        file_path (str): Path to the CSV file.
        full_gene_list (list of str): List of all gene names to extract.
        batch_label_list (list of str): List of batch label substrings to identify batches.
    
    Returns:
        np.ndarray or None: Combined feature array with shape (n_features,), or None if processing fails.
                           Features include gene expressions, perturbed gene indicators, and batch labels.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, index_col=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    # Extract gene expression data: average expression values for specified genes
    try:
        expression = df.loc[full_gene_list, df.columns[1:]].mean(axis=1).values.astype(np.float32)
    except KeyError as e:
        print(f"Missing genes in {file_path}: {e}")
        # Fill missing genes with zeros
        missing_genes = set(full_gene_list).difference(df.index)
        for gene in missing_genes:
            df.loc[gene] = 0.0
        expression = df.loc[full_gene_list, df.columns[1:]].mean(axis=1).values.astype(np.float32)
    
    # Extract perturbed gene labels: assuming 'cls' column exists
    if 'cls' in df.columns:
        perturbed_gene = df['cls'].values.astype(np.float32)
    else:
        # If 'cls' column doesn't exist, fill with zeros
        print(f"Missing 'cls' column in {file_path}")
        perturbed_gene = np.zeros(len(full_gene_list), dtype=np.float32)
    
    # Extract batch labels: based on whether filename contains batch label substrings
    basename = os.path.basename(file_path)
    batch = np.array([1.0 if substr in basename else 0.0 for substr in batch_label_list], dtype=np.float32)
    
    # Combine all features into a single array
    sample = np.concatenate([expression, perturbed_gene, batch])
    
    return sample

# -----------------------
# Main Function
# -----------------------
def main(mode, feature_dir, file_name):
    """
    Main function to process multiple CSV files and create a combined dataset.
    
    Args:
        mode (str): Processing mode - 'train' or 'eval'.
        feature_dir (str): Directory containing feature CSV files.
        file_name (str): Output filename for the processed data.
    
    Returns:
        None: Saves processed data as a .npy file.
    """
    # Get the intersection of files present in both feature and label directories
    basename_set = set([os.path.basename(i) for i in os.listdir('/public/home/shenninggroup/shenlab/tianyun/scCrisper_Data/scale_batchRemoval_0815/feature')]) & set(
        [os.path.basename(i) for i in os.listdir('/public/home/shenninggroup/shenlab/tianyun/scCrisper_Data/scale_batchRemoval_0815/label')]
    )
    # List of cell lines to include in the dataset
    cellline_list = ['crispr', 'cropseq', 'GSE102080', 'pancancer']
    
    # Get gene list and batch label list from helper functions
    full_gene_list = get_full_gene_list()
    batch_label_list = get_batch_label_list()
    
    # Collect all qualifying CSV files based on mode
    random.seed(1009)
    index_num = random.randint(1, 50)
    if mode == 'train':
        if cellline_list:
            # For training: exclude files with '_index_num' and include only specified cell lines
            feature_files = [
                f for f in sorted(glob.glob(os.path.join(feature_dir, '*.csv')))
                if (f'_{index_num}' not in f) and (os.path.basename(f) in basename_set)
                and any(substr in f for substr in cellline_list)
            ]
        else:
            # If no cell lines specified, include all files without '_index_num    '
            feature_files = [
                f for f in sorted(glob.glob(os.path.join(feature_dir, '*.csv')))
                if (f'_{index_num}' not in f) and (os.path.basename(f) in basename_set)
            ]
    
    num_files = len(feature_files)
    print(f"Total CSV files found: {num_files}")
    
    if num_files == 0:
        print("No files to process. Please check your filters.")
        return
    
    # Define number of worker processes (CPU cores - 1)
    num_workers = max(cpu_count() - 1, 1)
    print(f"Using {num_workers} workers for parallel processing.")
    
    # Create partial function with fixed gene list and batch label list
    if mode == 'train':
        partial_process = partial(process_csv, full_gene_list=full_gene_list, batch_label_list=batch_label_list)
    
    # Initialize storage list
    samples = []
    
    # Use multiprocessing pool for parallel processing
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for more efficient task processing
        results = pool.imap_unordered(partial_process, feature_files, chunksize=100)
        
        # Use tqdm to display progress bar
        for sample in tqdm(results, total=num_files, desc="Processing CSV files"):
            if sample is not None:
                samples.append(sample)
    
    num_samples = len(samples)
    print(f"Total valid samples processed: {num_samples}")
    
    if num_samples == 0:
        print("No valid samples were processed. Exiting.")
        return
    
    # Stack all samples into a 2D array
    data_array = np.stack(samples)  # Shape: [num_samples, feature_dim]
    
    # Save as .npy file
    output_dir = '/public/home/shenninggroup/lycxder/programme/cycle/code/CycleGAN4Seq/ijcai'
    np.save(os.path.join(output_dir, f'{file_name}.npy'), data_array)
    print(f"Data saved to {file_name} with shape {data_array.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CSV files and create a combined dataset.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Processing mode')
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory containing perturbed gene expression CSV files')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing control gene expression CSV files')
    parser.add_argument('--file_name', type=str, required=True, help='Output filename for the processed data')
    args = parser.parse_args()
    
    # Process evaluation data
    main(mode=args.mode, feature_dir=args.feature_dir, file_name=args.file_name)

