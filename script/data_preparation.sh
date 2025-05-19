#!/bin/bash
# Data Preparation Script for CycSeq
# This script downloads and organizes all necessary datasets for the CycSeq framework

echo "Starting data download and preparation process..."

# Create main data directory if it doesn't exist
mkdir -p data
cd data

# ==========================================
# GSE132080 Dataset
# ==========================================
echo "Downloading GSE132080 dataset..."
mkdir -p GSE102080
cd GSE102080

# Download files from NCBI GEO
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132080/suppl/GSE132080%5F10X%5Fbarcodes%2Etsv%2Egz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132080/suppl/GSE132080%5F10X%5Fgenes%2Etsv%2Egz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132080/suppl/GSE132080%5F10X%5Fmatrix%2Emtx%2Egz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132080/suppl/GSE132080%5Fcell%5Fidentities%2Ecsv%2Egz

echo "GSE132080 download complete."

# ==========================================
# CROP-seq NK Cell Cancer Cell Line Dataset
# ==========================================
echo "Downloading CROP-seq NK cell cancer cell line dataset..."
cd ../
mkdir -p cropseq
cd cropseq

# Note: Synapse requires authentication
echo "NOTE: The following command requires Synapse CLI and authentication."
echo "If not already logged in, you will be prompted for credentials."
echo "If you don't have a Synapse account, please register at https://www.synapse.org/"
synapse get -r syn52600685

# Download files from EBI
echo "Downloading CROP-seq files from EBI..."
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_K562_NK1_1_16_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_K562_NK1_1_16_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_K562_NK1_1_16_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_K562_noNK_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_K562_noNK_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_K562_noNK_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_LP1_NK1_1_4_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_LP1_NK1_1_4_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_LP1_NK1_1_4_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_LP1_NK1_1_16_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_LP1_NK1_1_16_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_LP1_NK1_1_16_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_LP1_noNK_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_LP1_noNK_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_LP1_noNK_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_4_3h_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_4_3h_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_4_3h_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_4_6h_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_4_6h_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_4_6h_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_4_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_4_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_4_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_16_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_16_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_NK1_1_16_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_activ_NK1_1_4_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_activ_NK1_1_4_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_activ_NK1_1_4_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_activ_NK1_1_16_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_activ_NK1_1_16_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_activ_NK1_1_16_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_activ_noNK_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_activ_noNK_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_activ_noNK_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_noNK_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_MM1S_noNK_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_NALM6_NK1_1_4_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_NALM6_NK1_1_4_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_NALM6_NK1_1_4_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_NALM6_NK1_1_16_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_NALM6_NK1_1_16_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_NALM6_NK1_1_16_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_NALM6_noNK_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_NALM6_noNK_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_NALM6_noNK_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_SUDHL4_NK1_1_4_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_SUDHL4_NK1_1_4_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_SUDHL4_NK1_1_4_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_SUDHL4_NK1_1_16_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_SUDHL4_NK1_1_16_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_SUDHL4_NK1_1_16_matrix.mtx.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_SUDHL4_noNK_barcodes.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_SUDHL4_noNK_features.tsv.gz	 
wget https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-13204/Files/CROPseq_SUDHL4_noNK_matrix.mtx.gz

echo "CROP-seq dataset download complete."

# ==========================================
# GSE133344 Dataset (Two-drug perturbation)
# ==========================================
echo "Downloading GSE133344 dataset (Two-drug perturbation)..."
cd ../
mkdir -p GSE133344
cd GSE133344

wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344%5Fraw%5Fbarcodes%2Etsv%2Egz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344%5Fraw%5Fcell%5Fidentities%2Ecsv%2Egz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344%5Fraw%5Fgenes%2Etsv%2Egz
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344%5Fraw%5Fmatrix%2Emtx%2Egz

echo "GSE133344 download complete."

# ==========================================
# CRISPR Datasets (Replogle et al.)
# ==========================================
echo "Setting up directories for CRISPR datasets (Replogle et al.)..."
cd ../
mkdir -p crispr
cd crispr

echo "MANUAL ACTION REQUIRED:"
echo "Please visit the following URL to download the Perturb-seq datasets:"
echo "https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387?file=35773075"
echo ""
echo "Download the following files and place them in the current directory ($(pwd)):"
echo "1. K562_essential_normalized_singlecell_01.h5ad"
echo "2. rpe1_normalized_singlecell_01.h5ad"
echo ""
echo "Press Enter when you have completed this step..."
read -p ""

# ==========================================
# Pan-Cancer Dataset
# ==========================================
echo "Setting up directories for Pan-Cancer dataset..."
cd ../
mkdir -p pancancer
cd pancancer

echo "MANUAL ACTION REQUIRED:"
echo "Please visit the following URL to download the Pan-Cancer cell line datasets:"
echo "https://singlecell.broadinstitute.org/single_cell/study/SCP542/pan-cancer-cell-line-heterogeneity#/"
echo ""
echo "You will need to create an account or log in to download the following files:"
echo "1. CPM_data.txt"
echo "2. Metadata.txt"
echo ""
echo "Download these files and place them in the current directory ($(pwd))."
echo "Press Enter when you have completed this step..."
read -p ""

# ==========================================
# Completion
# ==========================================
cd ../../
echo "Data preparation complete. All datasets have been downloaded to the 'data' directory."
echo "For datasets requiring manual download, please ensure you have followed the instructions."
echo "You can now proceed with the next steps of the CycSeq pipeline."