# io.py
import os
import glob
import shutil

def get_file_list(root_path, cellline, random_num=None):
    """
    Retrieve lists of feature and evaluation CSV files for a given cell line.
    
    Parameters:
        root_path (str): Base directory containing the 'feature' and 'label' subdirectories.
        cellline (str): Identifier used to filter files for a specific cell line.
        random_num (str, optional): Substring used for additional filtering of file names.
        
    Returns:
        tuple: (csv_files, eval_csv_files)
            csv_files (list): Full paths to feature CSV files matching the criteria.
            eval_csv_files (list): Full paths to evaluation CSV files (excluding files with 'neg').

    Raises:
        FileNotFoundError: If the root_path or required subdirectories do not exist.
        RuntimeError: If there is an error retrieving the CSV files.
    """
    # Validate the root path and necessary subdirectories.
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Root path '{root_path}' does not exist.")

    feature_path = os.path.join(root_path, 'feature')
    label_path = os.path.join(root_path, 'label')
    if not os.path.isdir(feature_path):
        raise FileNotFoundError(f"Feature directory '{feature_path}' does not exist.")
    if not os.path.isdir(label_path):
        raise FileNotFoundError(f"Label directory '{label_path}' does not exist.")
        
    print(f"Starting get_file_list for cellline: {cellline}")
    print(f"Searching in paths:\n - Feature: {feature_path}\n - Label: {label_path}")
    
    # Retrieve all CSV files from both directories.
    try:
        all_feature_files = glob.glob(os.path.join(feature_path, '*.csv'))
        all_label_files = glob.glob(os.path.join(label_path, '*.csv'))
    except Exception as e:
        raise RuntimeError(f"Error retrieving CSV files: {e}")
        
    print(f"Found {len(all_feature_files)} feature files and {len(all_label_files)} label files")
    
    # Define a helper filter function to check file criteria.
    def file_filter(filename):
        conditions = []
        if random_num is not None:
            conditions.append(random_num in filename)
        conditions.append('control' not in filename)
        conditions.append(cellline in filename)
        return all(conditions)
    
    # Filter files based on the criteria.
    feature_basenames = {os.path.basename(f) for f in all_feature_files if file_filter(f)}
    print(f"Found {len(feature_basenames)} matching feature files after filtering.")
    
    label_basenames = {os.path.basename(f) for f in all_label_files if file_filter(f)}
    print(f"Found {len(label_basenames)} matching label files after filtering.")
    
    # Determine common file basenames that exist in both directories.
    common_basenames = feature_basenames & label_basenames
    print(f"Found {len(common_basenames)} files common to both directories.")
    
    # Construct final file paths for the common files.
    csv_files = [f for f in all_feature_files if os.path.basename(f) in common_basenames]
    eval_csv_files = [f for f in all_label_files if os.path.basename(f) in common_basenames and 'neg' not in f]
    
    print(f"Final results:\n - Feature files: {len(csv_files)}\n - Eval files: {len(eval_csv_files)}")
    return csv_files, eval_csv_files

def move_files(source_dir, target_dir, key_word):
    """
    Move files matching a specified pattern from the source directory to the target directory.
    
    Parameters:
        source_dir (str): The directory to search for files.
        target_dir (str): The directory where matching files will be moved.
        key_word (str): The glob pattern to filter files.
        
    Returns:
        None
    """
    # Verify that the source directory exists
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Ensure the target directory exists; create it if necessary
    os.makedirs(target_dir, exist_ok=True)

    # Retrieve files matching the keyword pattern
    files_to_move = glob.glob(os.path.join(source_dir, key_word))
    if not files_to_move:
        print(f"No files matching pattern '{key_word}' found in {source_dir}")
    
    # Move each file to the target directory
    for file_path in files_to_move:
        try:
            shutil.move(file_path, target_dir)
        except Exception as e:
            print(f"Error moving file {file_path} to {target_dir}: {e}")
