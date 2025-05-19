# example processing code for CropSeq LP1 cell line.
library(Seurat)

seurat_object <- readRDS("./cropseq_lp1.rds")
seurat_object = UpdateSeuratObject(seurat_object)
seurat_object <- NormalizeData(seurat_object, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
seurat_object <- FindVariableFeatures(seurat_object, selection.method = "vst", nfeatures = 2000)
all.genes <- rownames(seurat_object)
seurat_object <- ScaleData(seurat_object, features= all.genes)
seurat_object_split <- SplitObject(seurat_object, split.by = "donor_id")



seurat_object_split_1 <- SplitObject(seurat_object_split$CROPseq_LP1_NK1_1_16, split.by = "gene")

for (i in c(unique(seurat_object_split$CROPseq_LP1_NK1_1_16@meta.data$gene))){
  file <- paste0("./NK1_16/",i, ".csv")
  print(i)
  write.csv(file=file,as.data.frame(seurat_object_split_1[[i]][["RNA"]]@scale.data),quote = F)
}

seurat_object <- LoadH5Seurat("./GSM5151370_PD213_scifi_2_CRISPR-TCR_77300_MeOH-cells.h5seurat")


for (i in c(unique(seurat_object@meta.data$Donor))){
  file <- paste0("./",i, ".csv")
  print(i)
  write.csv(file=file,as.data.frame(seurat_object_split[[i]][["RNA"]]@data),quote = F)
}