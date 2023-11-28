# Decipher-ATAC: extension of decipher to transcriptomics

![Screenshot 2023-11-20 at 10 25 00 AM](https://github.com/erialc-cal/decipher-ATAC/assets/61749901/7d31e498-bcac-4925-9d04-bab8269fa0ca){width=10%}

Runs similarly as for decipher but with two anndata datasets (h5ad) with corresponding cells: assume that `pdac_rna` is our gene expression data and `pdac_atac` our ATAC observations. 

```
from decipher.tools._decipher import DecipherConfig_withATAC
from decipher.tools.decipher import decipherATAC_train
```
Instantiate with `DecipherConfig_withATAC` and call `decipherATAC_train` to train model. 
```
config = DecipherConfig_withATAC(
    dim_genes = pdac_rna.shape[1],
    dim_atac= pdac_atac.shape[1])
decipherATAC_train(
    pdac_rna,
    pdac_atac,
    decipher_config=DecipherConfig_withATAC(),
    plot_every_k_epoch=1,
    plot_kwargs=None,
)
```
