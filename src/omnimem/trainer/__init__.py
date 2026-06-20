from .wan_ode_trainer import WanODETrainer
from .wan_ode_nsa_trainer import WanNSAODETrainer
from .wan_dmd_trainer import WanDMDTrainer
from .wan_self_forcing_trainer import WanSelfForcingTrainer
from .wan_nsa_self_forcing_trainer import WanNSASelfForcingTrainer
from .wan_nsa_self_forcing_streaming_trainer import WanNSASelfForcingStreamingTrainer
from .wan_nsa_self_forcing_streaming_switch_trainer import WanNSASelfForcingStreamingSwitchTrainer
from .wan_causal_trainer import WanCausalTrainer
from .wan_nsa_causal_trainer import WanCausalNSATrainer
from .wan_cd_trainer import WanConsistencyDistillTrainer
from .wan_nsa_cd_trainer import WanConsistencyDistillNASTrainer


TRAINER_CLS = {
    "dmd": WanDMDTrainer,
    "self-forcing": WanSelfForcingTrainer,
    "nsa-self-forcing": WanNSASelfForcingTrainer,
    "ode": WanODETrainer,
    "nsa-ode": WanNSAODETrainer,
    "nsa-self-forcing-switch": WanNSASelfForcingStreamingSwitchTrainer,
    "nsa-self-forcing-streaming": WanNSASelfForcingStreamingTrainer,
    "causal": WanCausalTrainer,
    "nsa-causal": WanCausalNSATrainer,
    "cd": WanConsistencyDistillTrainer,
    "nsa-cd": WanConsistencyDistillNASTrainer
}
