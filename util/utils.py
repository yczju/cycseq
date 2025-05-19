import os
import pandas as pd

def get_full_gene_list():
    """
    Retrieves a comprehensive list of gene names that are common across multiple datasets.
    
    This function reads gene names from various CSV files across different experimental conditions
    and cell lines, then finds the intersection of all these gene sets to create a unified gene list
    that is consistent across all datasets.
    
    The datasets include:
    - GSE132080 dataset
    - Various cancer cell lines (K562, LP1, MM1, NALM6, SUDHL4) with and without NK cells
    - CRISPR datasets for K562 essential genes and RPE1 cells
    - Two-drug perturbation dataset (GSE133344)
    - Pan-cancer scRNA dataset
    
    Returns:
        list: A sorted list of gene names that are present in all the specified datasets.
    """
    # Define root path for all data files
    root_path = 'data/'
    
    # Define paths for different datasets
    gse132080_path = os.path.join(root_path, 'GSE132080/gene_split_0515/ALDOA.csv')
    nk_cancer_base_path = os.path.join(root_path, 'cropseq_NKcell_CancerCellline')
    crispr_base_path = os.path.join(root_path, 'crispr')
    two_drug_path = os.path.join(root_path, 'GSE133344/COL2A1_NegCtrl0.csv')
    pancancer_path = os.path.join(root_path, 'scRNA_pancancer/2313287_STOMACH.csv')
    
    # Load gene lists from each dataset
    gene_sets = []
    
    # GSE132080 dataset
    gene_sets.append(set(pd.read_csv(gse132080_path, index_col=0).index.tolist()))
    
    # Cancer cell lines without NK cells
    cell_lines_no_nk = {
        'K562': 'B2M.csv',
        'LP1': 'ARID1A.csv',
        'MM1': 'AEBP2.csv',
        'NALM6': 'CASP8.csv',
        'SUDHL4': 'BID.csv'
    }
    
    for cell_line, file_name in cell_lines_no_nk.items():
        file_path = os.path.join(nk_cancer_base_path, cell_line, 'noNK', file_name)
        gene_sets.append(set(pd.read_csv(file_path, index_col=0).index.tolist()))
    
    # Cancer cell lines with NK cells (1:16 ratio)
    for cell_line, file_name in cell_lines_no_nk.items():
        file_path = os.path.join(nk_cancer_base_path, cell_line, 'NK1_16', file_name)
        gene_sets.append(set(pd.read_csv(file_path, index_col=0).index.tolist()))
    
    # Additional NK cell ratios for some cell lines (1:4 ratio)
    cell_lines_nk_1_4 = ['LP1', 'MM1', 'NALM6', 'SUDHL4']
    for cell_line in cell_lines_nk_1_4:
        file_name = cell_lines_no_nk[cell_line]
        file_path = os.path.join(nk_cancer_base_path, cell_line, 'NK1_4', file_name)
        gene_sets.append(set(pd.read_csv(file_path, index_col=0).index.tolist()))
    
    # CRISPR datasets
    gene_sets.append(set(pd.read_csv(os.path.join(crispr_base_path, 'K562_essential/AAAS.csv')).gene_name.tolist()))
    gene_sets.append(set(pd.read_csv(os.path.join(crispr_base_path, 'RPE1/AAAS.csv')).gene_name.tolist()))
    
    # Two-drug perturbation dataset
    gene_sets.append(set(pd.read_csv(two_drug_path, index_col=0).index.tolist()))
    
    # Pan-cancer scRNA dataset
    gene_sets.append(set(pd.read_csv(pancancer_path, index_col=0).index.tolist()))
    
    # Find intersection of all gene sets and sort
    full_gene_list = list(sorted(set.intersection(*gene_sets)))
    return full_gene_list

def get_batch_label_list(exclude = None):
    """
    Generates a list of batch labels for identifying different experimental batches in the dataset.
    
    This function creates a standardized list of batch identifiers that can be used to track
    the source of each data sample. The list includes various experimental conditions, cell lines,
    and dataset identifiers. It can optionally exclude certain categories of batches.
    
    Parameters:
        exclude (str, optional): Category of batches to exclude from the list.
            - If 'pancancer': Excludes pan-cancer dataset batches
            - If None: Includes all available batches
    
    Returns:
        list: A list of batch labels that can be used to identify the source of data samples.
            The list includes placeholder values to maintain consistent dimensionality.
    """
    # for new cell lines
    placeholder = ['placeholder1', 'placeholder2']
    if exclude == 'pancancer':
        batch_label_list = [
            'GSE102080', 'crispr_K562', 'crispr_RPE1',
            'cropseq_K562_NK1_16', 'cropseq_K562_noNK',
            'cropseq_LP1_NK1_16', 'cropseq_LP1_NK1_4', 'cropseq_LP1_noNK',
            'cropseq_MM1_NK1_16', 'cropseq_MM1_NK1_4',  'cropseq_MM1_noNK',
            'cropseq_NALM6_NK1_16', 'cropseq_NALM6_NK1_4', 'cropseq_NALM6_noNK',
            'cropseq_SUDHL4_NK1_16', 'cropseq_SUDHL4_NK1_4', 'cropseq_SUDHL4_noNK',
            ] + placeholder
    elif not exclude:
        batch_label_list = [
            'GSE102080', 'crispr_K562', 'crispr_RPE1',
            'cropseq_K562_NK1_16', 'cropseq_K562_noNK',
            'cropseq_LP1_NK1_16', 'cropseq_LP1_NK1_4', 'cropseq_LP1_noNK',
            'cropseq_MM1_NK1_16', 'cropseq_MM1_NK1_4',  'cropseq_MM1_noNK',
            'cropseq_NALM6_NK1_16', 'cropseq_NALM6_NK1_4', 'cropseq_NALM6_noNK',
            'cropseq_SUDHL4_NK1_16', 'cropseq_SUDHL4_NK1_4', 'cropseq_SUDHL4_noNK',
        ] + [f.replace('.csv', '') for f in sorted(os.listdir('/public/home/shenninggroup/shenlab/tianyun/scCrisper_Data/scRNA_pancancer')) if f.endswith('.csv')] + placeholder
    print(f'length of batch_label_list: {len(batch_label_list)}')
    return batch_label_list



def random_sampling(feature_path, output_feature_path, output_label_path, inner_gene_list, exclude_list: list = [], index_range: list = None):
    """
    Create a dataset by random sampling from feature files after perturbation.
    Afterwards, move control files to a specified directory.
    
    Parameters:
        feature_path (str): Directory containing the feature CSV files.
        output_feature_path (str): Directory to save the sampled feature CSV files.
        output_label_path (str): Directory to move control files.
        inner_gene_list (list): List of gene identifiers used for filtering.
        exclude_list (list, optional): List of filenames to exclude. Defaults to [].
        index_range (list, optional): Two-element list defining the subset range of files to process. Defaults to None.
        
    Returns:
        None
    """
    # Check if the feature directory exists
    if not os.path.isdir(feature_path):
        raise FileNotFoundError(f"Feature path not found: {feature_path}")
    
    # Ensure output directories exist
    os.makedirs(output_feature_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)

    # Obtain the list of files, applying index_range if provided
    all_files = sorted(os.listdir(feature_path))
    files_to_process = all_files[index_range[0]:index_range[1]] if index_range and len(index_range) == 2 else all_files

    for file in tqdm(files_to_process, desc="Random sampling for features"):
        if file.endswith('.csv') and file not in exclude_list:
            # Determine the position of the gene identifier in the file name based on naming conventions
            if '10K_K562' in file or 'GSE168620_Jurkat' in file or ('GSE90063_K562' in file and '_highMOI' not in file):
                position = -2
            elif '_highMOI' in file:
                position = -3
            else:
                position = -1

            try:
                identifier = file.replace('.csv', '').split('_')[position]
            except IndexError:
                print(f"Filename format error in file {file}")
                continue

            # Process the file if the identifier is in the inner_gene_list or if it's a control file
            if identifier in inner_gene_list or 'control' in file:
                if 'control' in file:
                    print(f"Processing control file: {file}")
                file_path = os.path.join(feature_path, file)
                try:
                    data = pd.read_csv(file_path, index_col=0)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue

                random_index = 0
                # Create 50 random samples of the feature data
                for file_index in range(1, 51):
                    try:
                        if data.shape[1] > 50:
                            data_random = data.sample(n=50, axis=1, random_state=2992 + random_index)
                        else:
                            data_random = data.sample(n=50, axis=1, random_state=2992 + random_index, replace=True)
                        # Insert a classification column 'cls': 1 if the gene identifier is found in the row index, otherwise 0
                        data_random.insert(0, 'cls', data_random.index.to_series().apply(lambda x: 1 if identifier in x else 0))
                        output_file = os.path.join(output_feature_path, file.replace('.csv', f'_{file_index}.csv'))
                        data_random.to_csv(output_file, index=True)
                    except Exception as e:
                        print(f"Error during sampling for file {file}, iteration {file_index}: {e}")
                    random_index += 1
        tqdm.write(f"Finished processing file: {file}")

    # After processing, move control files to the output label directory
    move_files(output_feature_path, output_label_path, '*control*.csv')