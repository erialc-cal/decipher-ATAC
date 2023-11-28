# Decipher-ATAC: extension of decipher to transcriptomics

 <img src="https://github.com/erialc-cal/decipher-ATAC/assets/61749901/7d31e498-bcac-4925-9d04-bab8269fa0ca" width="250" height="300">

Runs similarly as for decipher but with two anndata datasets (h5ad) with corresponding cells: assume that `rnadata` is our gene expression data and `atacdata` our ATAC observations. 

```
from decipher.tools._decipher import DecipherConfig_withATAC
from decipher.tools.decipher import decipherATAC_train
```
Instantiate with `DecipherConfig_withATAC` and call `decipherATAC_train` to train model. 
```
config = DecipherConfig_withATAC(
    dim_genes = rnadata.shape[1],
    dim_atac= atacdata.shape[1])
decipherATAC_train(
    rnadata,
    atacdata,
    decipher_config=DecipherConfig_withATAC(),
    plot_every_k_epoch=1,
    plot_kwargs=None,
)
```
Output of trajectories ($v_x, v_s$) on pdac data can be found in the folder `figs`. 
