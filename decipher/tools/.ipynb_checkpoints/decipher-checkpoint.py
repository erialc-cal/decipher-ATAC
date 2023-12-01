import logging

import numpy as np
import pyro
import scanpy as sc
import torch
from matplotlib import pyplot as plt
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import MultiStepLR
from tqdm import tqdm

from decipher.plot.decipher import decipher as plot_decipher_v
from decipher.tools._decipher import Decipher_with_ATAC, DecipherConfig_withATAC
from decipher.tools._decipher.data import (
    decipher_load_model,
    decipher_save_model,
    make_data_loader_from_tscp,
)
from decipher.tools.utils import EarlyStopping

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
)


def predictive_log_likelihood(decipher, dataloader):
    """Compute the predictive log likelihood of the decipher model."""
    if type(dataloader) == sc.AnnData:
        dataloader = make_data_loader_from_adata(dataloader, decipher.config.batch_size)

    decipher.eval()
    log_likelihood = 0
    for xc in dataloader:
        guide_trace = poutine.trace(decipher.guide).get_trace(*xc)
        replayed_model = poutine.replay(decipher.model, trace=guide_trace)
        blocked_replayed_model = poutine.block(replayed_model, expose=["x"])
        model_trace = poutine.trace(blocked_replayed_model).get_trace(*xc)
        log_likelihood += model_trace.log_prob_sum().item()
    return log_likelihood / len(dataloader.dataset)


def _make_train_val_split(adata, val_frac, seed):
    n_val = int(val_frac * adata.shape[0])
    cell_idx = np.arange(adata.shape[0])
    np.random.default_rng(seed).shuffle(cell_idx)
    val_idx = cell_idx[-n_val:]
    adata.obs["decipher_split"] = "train"
    adata.obs.loc[adata.obs.index[val_idx], "decipher_split"] = "validation"
    adata.obs["decipher_split"] = adata.obs["decipher_split"].astype("category")
    logging.info(
        "Added `.obs['decipher_split']`: the Decipher train/validation split.\n"
        f" {n_val} cells in validation set."
    )



def decipherATAC_train(
    rnadata: sc.AnnData,
    atacdata: sc.AnnData,
    decipher_config=DecipherConfig_withATAC(),
    plot_every_k_epoch=-1,
    plot_kwargs=None,
):
    """Train a decipher model with ATAC data and RNA data.

    Parameters
    ----------
    adata: sc.AnnData
        The annotated data matrix.
    decipher_config: DecipherConfig, optional
        Configuration for the decipher model.
    plot_every_k_epoch: int, optional
        If > 0, plot the decipher space every `plot_every_k_epoch` epochs.
        Default: -1 (no plots).
    plot_kwargs: dict, optional
        Additional keyword arguments to pass to `dc.pl.decipher`.

    Returns
    -------
    decipher: Decipher
        The trained decipher model.
    val_losses: list of float
        The validation losses at each epoch.
    `adata.obs['decipher_split']`: categorical
        The train/validation split.
    `adata.obsm['decipher_v']`: ndarray
        The decipher v space.
    `adata.obsm['decipher_z']`: ndarray
        The decipher z space.
    """
    pyro.clear_param_store()
    pyro.util.set_rng_seed(decipher_config.seed)

    _make_train_val_split(rnadata, decipher_config.val_frac, decipher_config.seed)
    train_idx = rnadata.obs["decipher_split"] == "train"
    val_idx = rnadata.obs["decipher_split"] == "validation"
    rnadata_train = rnadata[train_idx, :]
    rnadata_val = rnadata[val_idx, :]
    
    _make_train_val_split(atacdata, decipher_config.val_frac, decipher_config.seed)
    train_idx = atacdata.obs["decipher_split"] == "train"
    val_idx = atacdata.obs["decipher_split"] == "validation"
    atacdata_train = atacdata[train_idx, :]
    atacdata_val = atacdata[val_idx, :]
    
    if plot_kwargs is None:
        plot_kwargs = dict()

    decipher_config.initialize_from_adata(rnadata_train, atacdata_train)

    dataloader_train = make_data_loader_from_tscp(rnadata_train, atacdata_train, decipher_config.batch_size)
    dataloader_val = make_data_loader_from_tscp(rnadata_val, atacdata_val, decipher_config.batch_size)

    decipher = Decipher_with_ATAC(
        config=decipher_config,
    )
    optimizer = pyro.optim.ClippedAdam(
        {
            "lr": decipher_config.learning_rate,
            "weight_decay": 1e-4,
        }
    )
    elbo = Trace_ELBO()
    svi = SVI(decipher.model, decipher.guide, optimizer, elbo)
    # Training loop
    val_losses = []
    early_stopping = EarlyStopping(patience=5)
    pbar = tqdm(range(decipher_config.n_epochs))
    for epoch in pbar:
        train_losses = []
       
        decipher.train()
        if epoch > 0:
            # freeze the batch norm layers after the first epoch
            # 1) the batch norm layers helps with the initialization
            # 2) but then, they seem to imply a strong normal prior on the latent space
            for module in decipher.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.eval()

        for xc in dataloader_train:
            loss = svi.step(*xc)
            train_losses.append(loss)


        train_elbo = np.sum(train_losses) / len(dataloader_train.dataset) 
        decipher.eval()
        val_nll = -predictive_log_likelihood(decipher, dataloader_val)
        val_losses.append(val_nll)
        pbar.set_description(
            f"Epoch {epoch} | train elbo: {train_elbo:.2f} | val ll:" f" {val_nll:.2f}"
        )
        if early_stopping(val_nll):
            print("Early stopping.")
            break

        if plot_every_k_epoch > 0 and (epoch % plot_every_k_epoch == 0):
            _decipher_to_adata(decipher, rnadata, atacdata)
            plot_decipher_v(atacdata, basis="decipher_vs", **plot_kwargs)
            plot_decipher_v(rnadata, basis="decipher_vx", **plot_kwargs)
            plt.show()


    return decipher, val_losses


def _decipher_to_adata(decipher, rnadata, atacdata):
    """Compute the decipher v and z spaces from the decipher model. Add them to `adata.obsm`.

    Parameters
    ----------
    decipher: Decipher
        The decipher model.
    adata: sc.AnnData
        The annotated data matrix.

    Returns
    -------
    `adata.obsm['decipher_v']`: ndarray
        The decipher v space.
    `adata.obsm['decipher_z']`: ndarray
        The decipher z space.
    """
    decipher.eval()
    latent_zx, latent_zy, latent_zs, latent_vx, latent_vs = decipher.compute_v_z_numpy(rnadata.X.toarray(), atacdata.X.toarray())

    rnadata.obsm["decipher_vx"] = latent_vx
    atacdata.obsm["decipher_vs"] = latent_vs
    logging.info("Added `.obsm['decipher_v']`: the Decipher v space.")

    
    rnadata.obsm["decipher_zx"] = latent_zx
    atacdata.obsm["decipher_zs"] =latent_zs
    atacdata.obsm["decipher_zy"] = latent_zy
    logging.info("Added `.obsm['decipher_z']`: the Decipher z space.")

def _decipher_to_adata_var(decipher, rnadata, atacdata):
    """Compute the variances of decipher v and z spaces from the decipher model. Add them to `adata.obsm`.

    Parameters
    ----------
    decipher: Decipher
        The decipher model.
    adata: sc.AnnData
        The annotated data matrix.

    Returns
    -------
    `adata.obsm['decipher_v']`: ndarray
        The decipher v space.
    `adata.obsm['decipher_z']`: ndarray
        The decipher z space.
    """
    decipher.eval()
    latent_zx, latent_zy, latent_zs, latent_vx, latent_vs, vx_var, vs_var = decipher.compute_v_z_numpy(rnadata.X.toarray(), atacdata.X.toarray())

    rnadata.obsm["decipher_vx"] = latent_vx
    rnadata.obsm['vx_var'] = vx_var
    atacdata.obsm["decipher_vs"] = latent_vs
    atacdata.obsm['vs_var'] = vs_var
    logging.info("Added `.obsm['decipher_v']`: the Decipher v space.")

    
    rnadata.obsm["decipher_zx"] = latent_zx
    atacdata.obsm["decipher_zs"] =latent_zs
    atacdata.obsm["decipher_zy"] = latent_zy
    logging.info("Added `.obsm['decipher_z']`: the Decipher z space.")
