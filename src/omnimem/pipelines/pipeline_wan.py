# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import html
import re
import math
import urllib.parse as ul
import PIL
from contextlib import contextmanager
import numpy as np

import torch
import torch.cuda.amp as amp
from torchvision.transforms.v2.functional import to_tensor

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import (
    BACKENDS_MAPPING,
    is_bs4_available,
    is_ftfy_available,
)

from omnimem.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler

from typing import Optional

from omnimem.models.transformers.wan_model import WanModel
from omnimem.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from omnimem.pipelines.utils import masks_like
from omnimem.utils.train_utils import vae_encode
from omnimem.utils.logging_tool import get_logger

from omnimem.models.text_encoders.t5 import T5EncoderModel

logger = get_logger()

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class WanTI2VPipeline(DiffusionPipeline):
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
            self,
            text_encoder: T5EncoderModel,
            vae: AutoencoderKLWan,
            transformer: WanModel,
            tokenizer: None,
            scheduler: FlowUniPCMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.image_processor = None

        self.num_train_timesteps = 1000
        self.param_dtype = torch.bfloat16

        self.vae_stride = (
            self.vae.temporal_downscale_factor,
            self.vae.spatial_downscale_factor,
            self.vae.spatial_downscale_factor
        )
        self.patch_size = (1, 2, 2)

        self.sp_size = 1

    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            logger.warn(
                BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`")
            )
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            logger.warn(
                BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`")
            )
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
        )

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(
            self.bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
        )
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
        )  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    @staticmethod
    def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "pil"):
        batch_size, channels, num_frames, height, width = video.shape
        outputs = []
        for batch_idx in range(batch_size):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3).clamp(-1, 1)
            batch_vid = batch_vid * 127.5 + 127.5
            batch_output = batch_vid.to(torch.uint8)
            if not output_type == "pt":
                batch_output = batch_output.numpy()
                if output_type == "pil":
                    batch_output = [PIL.Image.fromarray(output.transpose(1, 2, 0)) for output in batch_output]
            outputs.append(batch_output)

        if output_type == "np":
            outputs = np.stack(outputs)

        elif output_type == "pt":
            outputs = torch.stack(outputs)

        elif not output_type == "pil":
            raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

        return outputs

    def generate(
            self,
            input_prompt,
            size=(1280, 720),
            frame_num=81,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            generator=None,
            offload_model=False,
            img: Optional[PIL.Image.Image] = None
    ):
        """Generate video frames from a text prompt.

        Args:
            input_prompt: Text prompt.
            size: Video resolution (width, height).
            frame_num: Number of frames (should be 4n+1).
            shift: Noise schedule shift.
            sample_solver: Solver type.
            sampling_steps: Number of denoising steps.
            guide_scale: Classifier-free guidance scale.
            n_prompt: Negative prompt.
            generator: Torch random generator.
            offload_model: Offload models to CPU to save VRAM.
            img: Optional conditioning image for I2V.
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        input_prompt = self._text_preprocessing(input_prompt)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        self.text_encoder.model.to(self.device)
        context, attn_cond = self.text_encoder(input_prompt, self.device)
        context_null, attn_null = self.text_encoder(n_prompt, self.device)
        if offload_model:
            self.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        # prepare image condition
        if img is not None:
            assert isinstance(img, PIL.Image.Image), f"{img.type=} is not pillow image"
            # reshape
            w, h = img.size
            target_w, target_h = size
            assert w <= target_w, f"{w=} > {size[0]=}"
            assert h <= target_h, f"{h=} > {size[1]=}"
            if w != target_w or h != target_h:
                img = img.resize(size)
            img_tensor = to_tensor(img)[None, :, None] * 2 - 1
            self.vae.encoder.to(self.device)
            img_tensor = img_tensor.to(device=self.device, dtype=self.param_dtype)
            img_latent = vae_encode(vae=self.vae, pixel_values=img_tensor)
            if offload_model:
                self.vae.encoder.cpu()
                torch.cuda.empty_cache()
        else:
            img_latent = None

        noise = torch.randn(
            1,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=generator)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.transformer, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=True)

            if img_latent is not None:
                # replace first frame to image
                latents = (1. - mask1) * img_latent + mask1 * latents

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            with self.progress_bar(total=sampling_steps) as progress_bar:
                for _, t in enumerate(timesteps):
                    latent_model_input = latents
                    timestep = [t]

                    timestep = torch.stack(timestep)
                    if img_latent is not None:
                        timestep = mask1[:, 0, :, ::self.patch_size[1], ::self.patch_size[2]] * timestep

                    self.transformer.to(self.device)
                    noise_pred_cond = self.transformer(
                        latent_model_input, t=timestep, **arg_c)
                    if guide_scale > 1:
                        noise_pred_uncond = self.transformer(
                            latent_model_input, t=timestep, **arg_null)

                        noise_pred = noise_pred_uncond + guide_scale * (
                                noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=generator)[0]
                    latents = temp_x0.squeeze(0)
                    if img_latent is not None:
                        latents = (1. - mask2) * img_latent + mask2 * latents

                    progress_bar.update()
            if offload_model:
                self.transformer.cpu()
                torch.cuda.empty_cache()

            latents_mean = torch.tensor(self.vae.config.latents_mean,
                                        device=latents.device, dtype=latents.dtype).reshape(1, -1, 1, 1, 1)
            latents_std = torch.tensor(self.vae.config.latents_std,
                                       device=latents.device, dtype=latents.dtype).reshape(1, -1, 1, 1, 1)
            latents = latents * latents_std + latents_mean

            image = self.vae.decode(latents).sample
            image = self.tensor2vid(image.float().cpu(), self.image_processor, output_type="pil")

            if offload_model:
                self.transformer.cuda()

            return ImagePipelineOutput(images=image)
