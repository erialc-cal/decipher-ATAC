import dataclasses
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
import torch.utils.data
from torch.distributions import constraints
from torch.nn.functional import softmax, softplus

from decipher.tools._decipher.module import ConditionalDenseNN
from decipher.tools._decipher.module import ConditionalDenseNN

from decipher.tools._decipher.module import ConditionalDenseNN


@dataclass(unsafe_hash=True)
class DecipherConfig_withATAC:
    dim_z: int = 10
    dim_v: int = 2
    layers_v_to_z: Sequence = (64,)
    layers_z_to_x: Sequence = tuple()
    layers_z_to_y: Sequence = tuple()

    beta: float = 1e-1
    seed: int = 0

    learning_rate: float = 5e-3
    val_frac: float = 0.1
    batch_size: int = 64
    n_epochs: int = 100

    dim_genes: int = None
    dim_atac: int = None 
    n_cells: int = None
    prior: str = "normal"

    _initialized_from_adata: bool = False

    def initialize_from_adata(self, rnadata, atacdata):
        self.dim_genes = rnadata.shape[1]
        self.dim_atac = atacdata.shape[1]
        self.n_cells = rnadata.shape[0]
        self._initialized_from_adata = True

    def to_dict(self):
        res = dataclasses.asdict(self)
        res["layers_v_to_z"] = list(res["layers_v_to_z"])
        res["layers_z_to_x"] = list(res["layers_z_to_x"])
        res["layers_z_to_y"] = list(res["layers_z_to_y"])
        return res


class Decipher_with_ATAC(nn.Module):
    """Decipher _decipher for single-cell data.

    Parameters
    ----------
    config : DecipherConfig or dict
        Configuration for the decipher _decipher.
    """

    def __init__(
        self,
        config: Union[DecipherConfig_withATAC, dict] = DecipherConfig_withATAC(),
    ):
        super().__init__()
        if type(config) == dict:
            config = DecipherConfig_withATAC(**config)

        if not config._initialized_from_adata:
            raise ValueError(
                "DecipherConfig must be initialized from an AnnData object, "
                "use `DecipherConfig.initialize_from_adata(adata)` to do so."
            )

        self.config = config
        ### DECODERS
        # vx ---> zx, vs ----> zs, vs ----> zy
        self.decoder_vx_to_zx = ConditionalDenseNN(
            self.config.dim_v, self.config.layers_v_to_z, [self.config.dim_z] * 2
        )
        
        self.decoder_vy_to_zy = ConditionalDenseNN(
            self.config.dim_v, self.config.layers_v_to_z, [self.config.dim_z] * 2
        )
        # change vs ---> zs
        self.decoder_vs_to_zs = ConditionalDenseNN(
            self.config.dim_v, self.config.layers_v_to_z, [self.config.dim_z] * 2
        )
        
        
        # zx ---> x <---- zs, zy ----> y 
        self.decoder_zxzs_to_x = ConditionalDenseNN(
            2* self.config.dim_z, config.layers_z_to_x, [self.config.dim_genes]
        )
        self.decoder_zy_to_y = ConditionalDenseNN(
            self.config.dim_z, config.layers_z_to_y, [self.config.dim_atac]
        )
        
        
        
        ### ENCODERS
        #  zx ---> x <---- zs, zy ----> y 
        self.encoder_x_to_zxzs = ConditionalDenseNN(
            self.config.dim_genes, [128], [self.config.dim_z * 2] * 2 
        )
        self.encoder_y_to_zy = ConditionalDenseNN(
            self.config.dim_atac, [128], [self.config.dim_z] * 2
        )
        
        #  vx ----> zx, zs <----- vs ----> zy
        self.encoder_zx_to_v = ConditionalDenseNN(
            self.config.dim_genes +  self.config.dim_z, [128], [self.config.dim_v, self.config.dim_v]
        )

        self.encoder_zsy_to_v = ConditionalDenseNN(
            self.config.dim_atac +  2* self.config.dim_z, [128], [self.config.dim_v, self.config.dim_v]
        )

        self._epsilon = 1e-5

        self.theta = None
        self.eta = None
        print("V5")

    def model(self, x, y, context=None):
        """#define the model p(x|z)p(z) """
        pyro.module("decipher", self)
        ## Initialisation
        
        ### geX dispersion
        self.theta = pyro.param(
            "inverse_dispersion_gex",
            1.0 * x.new_ones(self.config.dim_genes),
            constraint=constraints.positive,
        )
        ### ATAC dispersion
        self.eta = pyro.param(
            "inverse_dispersion_atac",
            1.0 * y.new_ones(self.config.dim_atac),
            constraint=constraints.positive,
        )
 
        ## Sampling 
        assert len(x) == len(y), "cell numbers match"
        
      
        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            
            ### geX variables
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "normal":
                    prior = dist.Normal(0, x.new_ones(self.config.dim_v)).to_event(1)
                elif self.config.prior == "gamma":
                    prior = dist.Gamma(0.3, x.new_ones(self.config.dim_v) * 0.8).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                vx = pyro.sample("vx", prior)
        
            ### ATAC variables 
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "normal":
                    prior_y = dist.Normal(0, y.new_ones(self.config.dim_v)).to_event(1)
                elif self.config.prior == "gamma":
                    prior_y = dist.Gamma(0.3, y.new_ones(self.config.dim_v) * 0.8).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                vs = pyro.sample("vs", prior_y)

            ### geX variables    
            #vsx = torch.cat([vs,vx],dim=-1)
            zs_loc, zs_scale = self.decoder_vs_to_zs(vs, context=context)

            zs_scale = softplus(zs_scale)
            zs = pyro.sample("zs", dist.Normal(zs_loc, zs_scale).to_event(1))

            zx_loc, zx_scale = self.decoder_vx_to_zx(vx, context=context)
            zx_scale = softplus(zx_scale)
            zx = pyro.sample("zx", dist.Normal(zx_loc, zx_scale).to_event(1))                 
            zxs = torch.cat([zx,zs],dim=-1)

            mu_x = self.decoder_zxzs_to_x(zxs, context=context)
            mu_x = softmax(mu_x, dim=-1)
            library_size_x = x.sum(axis=-1, keepdim=True)
            # Parametrization of Negative Binomial by the mean and inverse dispersion
            # See https://github.com/pytorch/pytorch/issues/42449
            # noinspection PyTypeChecker
            logit_x = torch.log(library_size_x * mu_x + self._epsilon) - torch.log(
                self.theta + self._epsilon
            )
            
            # noinspection PyUnresolvedReferences
            x_dist = dist.NegativeBinomial(total_count=self.theta, logits=logit_x)
            pyro.sample("x", x_dist.to_event(1), obs=x)
            
             ### ATAC variables 
            zy_loc, zy_scale = self.decoder_vy_to_zy(vs, context=context)
            zy_scale = softplus(zy_scale)
            zy = pyro.sample("zy", dist.Normal(zy_loc, zy_scale).to_event(1))
            mu_y = self.decoder_zy_to_y(zy, context=context)
            mu_y = softmax(mu_y, dim=-1)
            library_size_y = y.sum(axis=-1, keepdim=True)
            
            logit_y = torch.log(library_size_y * mu_y + self._epsilon) - torch.log(
                self.eta + self._epsilon
            )
            
            """ADDED CHECK""" 
           # print("Neg Bin parameters", self.theta, logit) 
            
            
            # noinspection PyUnresolvedReferences
            y_dist = dist.NegativeBinomial(total_count=self.eta, logits=logit_y)
            pyro.sample("y", y_dist.to_event(1), obs=y)
            
            
    def guide(self, x, y, context=None):
        """ define the guide (i.e. variational distribution) q(z|x)
        """
    
        assert len(x) == len(y), "cell numbers match"
        pyro.module("decipher", self)
        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            
            ## geX guide
            x = torch.log1p(x)

            zjoint_loc, zjoint_scale = self.encoder_x_to_zxzs(x, context=context)
            zjoint_scale = softplus(zjoint_scale)
            posterior_zjoint = dist.Normal(zjoint_loc, zjoint_scale).to_event(1)
            zjoint = pyro.sample("(zx,zs)", posterior_zjoint, infer={'is_auxiliary': True})
            
            zs_loc = zjoint_loc[:,:self.config.dim_z]
            zs_scale = zjoint_scale[:,:self.config.dim_z]
            posterior_zs = dist.Normal(zs_loc, zs_scale).to_event(1)
            pyro.sample("zs", posterior_zs) 
            
            zx_loc = zjoint_loc[:,self.config.dim_z:]
            zx_scale = zjoint_scale[:,self.config.dim_z:]
            posterior_zx = dist.Normal(zx_loc, zx_scale).to_event(1)
            pyro.sample("zx", posterior_zx) 
            
            zx = zjoint[:,self.config.dim_z:]
            zs = zjoint[:,:self.config.dim_z]
               
            zx = torch.cat([zx, x], dim=-1)
            
            
            vx_loc, vx_scale = self.encoder_zx_to_v(zx, context=context)
            vx_scale = softplus(vx_scale)
            
            
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "gamma":
                    posterior_vx = dist.Gamma(softplus(vx_loc), vx_scale).to_event(1)
                elif self.config.prior == "normal" or self.config.prior == "student-normal":
                    posterior_vx = dist.Normal(vx_loc, vx_scale).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                pyro.sample("vx", posterior_vx)

            ## ATAC guide
        
            y = torch.log1p(y)

            zy_loc, zy_scale = self.encoder_y_to_zy(y, context=context)
            zy_scale = softplus(zy_scale)
            posterior_zy = dist.Normal(zy_loc, zy_scale).to_event(1)
            zy = pyro.sample("zy", posterior_zy)
            
            zyy = torch.cat([zs, zy, y], dim=-1)
            
            vs_loc, vs_scale = self.encoder_zsy_to_v(zyy, context=context)
            vs_scale = softplus(vs_scale)
            
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "gamma":
                    posterior_vs = dist.Gamma(softplus(vs_loc), vs_scale).to_event(1)
                elif self.config.prior == "normal" or self.config.prior == "student-normal":
                    posterior_vs = dist.Normal(vs_loc, vs_scale).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                vs = pyro.sample("vs", posterior_vs)
                
        return zx_loc, zy_loc, zjoint_loc, vx_loc, vs_loc, zx_scale, zy_scale, zjoint_scale, vx_scale, vs_scale

    def compute_v_z_numpy(self, x: np.array, y:np.array):
        """Compute decipher_vs and decipher_zs for a given input.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (n_cells, n_genes).
        y : np.ndarray or torch.Tensor
            Input data of shape (n_cells, n_atac).
        Returns
        -------
        vs : np.ndarray
            Decipher components v of shape (n_cells, dim_v).
        zs : np.ndarray
            Decipher latent z of shape (n_cells, dim_z).
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
        if type(y) == np.ndarray:
            y = torch.tensor(y, dtype=torch.float32)
            
        x = torch.log1p(x)
        y = torch.log1p(y)
        
        zjoint_loc, _ = self.encoder_x_to_zxzs(x)
        zs_loc =  zjoint_loc[:,:self.config.dim_z]
        zx_loc = zjoint_loc[:,self.config.dim_z:]
        zy_loc, _ = self.encoder_y_to_zy(y)
        zsy = torch.cat([zs_loc, zy_loc, y], dim=-1)
        zx = torch.cat([zx_loc,x],dim=-1)
        
        vs_loc, _ = self.encoder_zsy_to_v(zsy)
        vx_loc, _ = self.encoder_zx_to_v(zx)
    
        return  zx_loc.detach().numpy(), zy_loc.detach().numpy(), zs_loc.detach().numpy(), vx_loc.detach().numpy(), vs_loc.detach().numpy()
 
    def impute_expression_numpy(self, x, y):
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
        if type(y) == np.ndarray:
            y = torch.tensor(y, dtype=torch.float32)
        _, zy_loc, zjoint_loc, _, _, _, _, _, _, _ = self.guide(x, y)
        mux = self.decoder_zx_zs_to_x(zjoint_loc)
        mux = softmax(mux, dim=-1)
        library_size_x = x.sum(axis=-1, keepdim=True)
        muy = self.decoder_zy_to_y(zy_loc)
        muy = softmax(muy, dim=-1)
        library_size_y = y.sum(axis=-1, keepdim=True)
        
        return (library_size_x * mux).detach().numpy(), (library_size_y * muy).detach().numpy()
    
