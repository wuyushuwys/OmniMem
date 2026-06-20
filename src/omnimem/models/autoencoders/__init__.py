from .autoencoder_kl_wan import AutoencoderKLWan, WanDecoder3d, WanEncoder3d
from diffusers.models.autoencoders import AutoencoderKL

AUTOENCODERS = {
    "AutoencoderKLWan": AutoencoderKLWan,
}

MAXIMIZE_VAE_CKPT = (
    WanDecoder3d,
    WanEncoder3d,
)

AUTOENCODER_MODULES = (
    WanDecoder3d,
    WanEncoder3d,
)


def get_autoencoder(pretrained_model_name_or_path):
    if pretrained_model_name_or_path.endswith('.json'):
        import json
        vae_cls: str = json.load(open(pretrained_model_name_or_path, 'r'))["_class_name"]
    else:
        vae_cls: str = AutoencoderKL.load_config(pretrained_model_name_or_path)["_class_name"]
    vae_cls = vae_cls.removeprefix("FSDP")
    if pretrained_model_name_or_path.endswith('.json'):
        import json
        return AUTOENCODERS[vae_cls].from_config(json.load(open(pretrained_model_name_or_path, mode='r')))
    else:
        return AUTOENCODERS[vae_cls].from_pretrained(pretrained_model_name_or_path)
