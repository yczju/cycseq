# harmony.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

def batch_removal(df_ref, ref_list, inner_data_path, batch_label, gene_list, exclude_list: list = []):
    """
    Merge batch CSV files with reference data and perform batch correction using pycombat.
    
    Parameters:
        df_ref (DataFrame): The reference DataFrame.
        ref_list (list): List of reference sample labels.
        inner_data_path (str): Directory path containing batch CSV files.
        batch_label (str): Label identifier for the batch.
        gene_list (list): List of genes to include in the dataset.
        exclude_list (list, optional): List of filenames to exclude. Defaults to [].
        
    Returns:
        tuple: A tuple containing:
            - DataFrame: Adjusted batch data after correction.
            - list: Corresponding list of filenames for each data column.
    """
    # Validate that the provided directory exists
    if not os.path.isdir(inner_data_path):
        raise FileNotFoundError(f"Data directory not found: {inner_data_path}")

    # Initialize an empty DataFrame with gene_list as its index
    data_merge = pd.DataFrame(index=gene_list)
    batch_list = []  # List to store batch labels corresponding to each column
    inner_filename_list = []  # List to store filenames corresponding to each column

    try:
        # Filter CSV files, excluding those in the exclude_list
        file_list = [f for f in os.listdir(inner_data_path) if f.endswith('.csv') and f not in exclude_list]
    except Exception as e:
        raise Exception(f"Error listing files in directory {inner_data_path}: {e}")

    # Process each CSV file with a progress bar
    for file in tqdm(file_list, desc="Processing batch files"):
        file_path = os.path.join(inner_data_path, file)
        try:
            # Process files with "crispr" differently due to distinct format
            if 'crispr' in batch_label:
                data_temp = pd.read_csv(file_path)
                data_temp.drop_duplicates(subset=['gene_name'], keep='first', inplace=True)
                data_temp.set_index('gene_name', inplace=True)
            else:
                data_temp = pd.read_csv(file_path, index_col=0)
                # For genes missing in the current file, add rows with zeros
                missing_genes = set(gene_list) - set(data_temp.index)
                for gene in missing_genes:
                    data_temp.loc[gene, :] = 0
                # Reorder rows to follow the gene_list order
                data_temp = data_temp.loc[gene_list]
            # Concatenate the current file's data with the merged DataFrame
            data_merge = pd.concat([data_merge, data_temp], axis=1)
            num_columns = data_temp.shape[1]
            batch_list.extend([batch_label] * num_columns)
            inner_filename_list.extend([file] * num_columns)
        except Exception as e:
            print(f"Warning: Failed to process file {file}. Error: {e}")

    # Combine reference data with merged batch data and apply batch correction using pycombat
    try:
        combined_data = pd.concat([df_ref, data_merge], axis=1)
        corrected_data = pycombat(combined_data, pd.Series(ref_list + batch_list))
    except Exception as e:
        raise Exception(f"Batch correction using pycombat failed: {e}")

    # Return only the adjusted batch data (excluding reference columns) and the corresponding filename list
    return corrected_data.iloc[:, df_ref.shape[1]:], inner_filename_list

def get_batch_removal_dataset(adjusted_data, filename_list, output_feature_path, prefix):
    """
    Export the batch removal results by splitting the adjusted data based on batch filename frequency.
    
    Parameters:
        adjusted_data (DataFrame): The batch-corrected data.
        filename_list (list): List of filenames corresponding to the data columns.
        output_feature_path (str): Directory where the output CSV files will be saved.
        prefix (str): Filename prefix to use for the exported files.
        
    Returns:
        None
    """
    # Ensure the output directory exists; if not, create it
    os.makedirs(output_feature_path, exist_ok=True)
    
    frequency_dict = Counter(filename_list)
    data_copy = adjusted_data.copy()

    # Export each batch's data segment into separate CSV files
    for batch, freq in frequency_dict.items():
        output_file = os.path.join(output_feature_path, f'{prefix}_{batch}.csv')
        try:
            data_slice = data_copy.iloc[:, :freq]
            data_slice.to_csv(output_file, index=True)
        except Exception as e:
            print(f"Error writing file {output_file}: {e}")
        # Remove the exported columns from the copy for the next iteration
        data_copy = data_copy.iloc[:, freq:]

