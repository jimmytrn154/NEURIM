"""SDXL-Turbo / LCM wrapper: latent (well, prompt-embedding) in, frame out in
~100-300ms. Everything torch/diffusers is lazy-imported so the rest of the
codebase runs fine on a machine with no GPU - see procedural.py for what
runs instead in that case.
"""

from __future__ import annotations

import numpy as np


class DiffusionGenerator:
    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-turbo",
        num_inference_steps: int = 2,
        device: str | None = None,
    ):
        import torch
        from diffusers import AutoPipelineForText2Image

        self.device = device or self._detect_device(torch)
        self.num_inference_steps = num_inference_steps
        self._torch = torch

        if self.device == "cuda":
            self.dtype = torch.float16
            load_kwargs = {"torch_dtype": self.dtype, "variant": "fp16"}
        else:
            self.dtype = torch.float32
            load_kwargs = {"torch_dtype": self.dtype}

        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, **load_kwargs).to(self.device)
        self._prev_image = None
        # Captured the first time encode_prompts() runs, so render_from_embedding()
        # can split a flattened embedding back into SDXL's (prompt_embeds,
        # pooled_prompt_embeds) pair with the right shapes.
        self._prompt_embed_shape: tuple[int, int] | None = None  # (seq_len, hidden)
        self._pooled_dim: int | None = None

    @staticmethod
    def _detect_device(torch) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _generator(self, seed: int | None):
        """A fixed-seed torch.Generator so nearby embeddings render to nearby
        images - essential for the morph: without a pinned seed each frame
        re-rolls the base noise and the sequence flickers instead of morphing.
        """
        if seed is None:
            return None
        return self._torch.Generator(device=self.device).manual_seed(int(seed))

    def render_from_prompt(self, prompt: str, seed: int | None = None):
        """Straight text -> image, bypassing the embedding/projector path."""
        image = self.pipe(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            generator=self._generator(seed),
        ).images[0]
        self._prev_image = image
        return image

    def encode_prompts(self, prompts: list[str]) -> np.ndarray:
        """Encode each anchor prompt and return, per prompt, a single flat vector
        holding SDXL's full sequence embedding concatenated with its pooled
        embedding: [prompt_embeds.flatten(), pooled_prompt_embeds.flatten()].
        The projector is fit over these; render_from_embedding() reverses it.
        """
        embeddings = []
        for prompt in prompts:
            with self._torch.no_grad():
                out = self.pipe.encode_prompt(
                    prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                )
            # SDXL returns a 4-tuple; the pooled embed (3rd element) is required
            # by the UNet and must not be dropped. SD1.x returns just prompt_embeds.
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                prompt_embeds, pooled = out[0], out[2]
            else:
                prompt_embeds = out[0] if isinstance(out, (tuple, list)) else out
                pooled = None

            seq_len, hidden = prompt_embeds.shape[-2], prompt_embeds.shape[-1]
            self._prompt_embed_shape = (int(seq_len), int(hidden))
            flat = prompt_embeds.flatten().cpu().numpy()
            if pooled is not None:
                self._pooled_dim = int(pooled.shape[-1])
                flat = np.concatenate([flat, pooled.flatten().cpu().numpy()])
            embeddings.append(flat)
        return np.stack(embeddings)

    def render_from_embedding(self, embedding: np.ndarray, seed: int | None = None):
        """Reconstruct (prompt_embeds, pooled_prompt_embeds) from a flat vector
        produced by the projector (which was fit on encode_prompts() output) and
        render. This is the continuous path the latent morph rides on.
        """
        assert self._prompt_embed_shape is not None, "call encode_prompts() before render_from_embedding()"
        seq_len, hidden = self._prompt_embed_shape
        n_seq = seq_len * hidden

        vec = self._torch.tensor(embedding, dtype=self.dtype, device=self.device)
        prompt_embeds = vec[:n_seq].reshape(1, seq_len, hidden)

        kwargs = {}
        if self._pooled_dim is not None:
            pooled = vec[n_seq : n_seq + self._pooled_dim].reshape(1, self._pooled_dim)
            kwargs["pooled_prompt_embeds"] = pooled

        image = self.pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            generator=self._generator(seed),
            **kwargs,
        ).images[0]
        self._prev_image = image
        return image

    # Back-compat alias: older callers referenced .render()/.render_prompt().
    def render(self, embedding: np.ndarray, seed: int | None = None):
        return self.render_from_embedding(embedding, seed=seed)

    def render_prompt(self, prompt: str, seed: int | None = None):
        return self.render_from_prompt(prompt, seed=seed)

    def render_smoothed(self, embedding: np.ndarray, strength: float = 0.2, seed: int | None = None):
        """img2img pass against the previous frame, for smoother morphing
        between optimizer steps than independent from-scratch samples.
        """
        if self._prev_image is None:
            return self.render_from_embedding(embedding, seed=seed)
        from diffusers import AutoPipelineForImage2Image

        assert self._prompt_embed_shape is not None, "call encode_prompts() before render_smoothed()"
        seq_len, hidden = self._prompt_embed_shape
        n_seq = seq_len * hidden

        if not hasattr(self, "_img2img_pipe"):
            self._img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
        vec = self._torch.tensor(embedding, dtype=self.dtype, device=self.device)
        prompt_embeds = vec[:n_seq].reshape(1, seq_len, hidden)

        kwargs = {}
        if self._pooled_dim is not None:
            kwargs["pooled_prompt_embeds"] = vec[n_seq : n_seq + self._pooled_dim].reshape(1, self._pooled_dim)

        image = self._img2img_pipe(
            image=self._prev_image,
            prompt_embeds=prompt_embeds,
            strength=strength,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            generator=self._generator(seed),
            **kwargs,
        ).images[0]
        self._prev_image = image
        return image
