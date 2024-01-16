from modules import shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger
import hydra
import omegaconf
import wandb
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from modules.text_generation import get_max_prompt_length
from pathlib import Path
import json


class SlalomModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path):
        result = cls()
        model_dir = Path(path).parent
        try:
            with open((model_dir / "config.json")) as f:
                slalom_cfg = json.load(f)
        except Exception:
            slalom_cfg = {}
        tokenizer_name = slalom_cfg.get("tokenizer", "EleutherAI/pythia-160m-v0")
        base_job = slalom_cfg.get("base_job", "slalom/hologram-train/ryoi69ht")
        api = wandb.Api()
        run = api.run(base_job)
        cfg = omegaconf.DictConfig(run.config)
        if path.suffix == ".json":
            logger.info(f"No ckpt found in {model_dir}, need to download it first")
            project, job_id = base_job.rsplit("/", 1)
            artifact = api.artifact(
                "/".join(project, f"model-{job_id}:best")
            )  # assumes `best` exists
            # uses cached version if artifact is already downloaded
            ckpt_path = artifact.download() + "/model.ckpt"
        else:
            ckpt_path = path

        # TODO: support tweaking params from webui, support more models.
        result.model = hydra.utils.instantiate(cfg.model)
        ckpt = torch.load(ckpt_path)

        # Hack to transform state dict of LightningModule into nn.Module
        def f(x):
            return x.removeprefix("model.")

        result.model.load_state_dict(
            {f(k): v for k, v in ckpt["state_dict"].items()}, strict=False
        )
        result.model.to("cuda")
        logger.info(f"Using slalom model_type: {str(result.model)} for {ckpt_path}")

        kwargs = (
            {"decode_with_prefix_space": True}
            if "llama" in tokenizer_name.lower()
            else {}
        )  # hack for llama
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token="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",
            use_fast=True,
            **kwargs,
        )
        result.tokenizer = tokenizer
        return result, tokenizer

    def model_type_is_auto(self):
        return (
            shared.args.model_type is None
            or shared.args.model_type == "Auto"
            or shared.args.model_type == "None"
        )

    def model_dir(self, path):
        if path.is_file():
            return path.parent

        return path

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def generate(self, prompt, state, callback=None):
        prompt = prompt if isinstance(prompt, str) else prompt.decode()
        models_device = next(self.model.parameters()).device
        # Handle truncation
        prompt = self.encode(prompt)
        prompt = prompt[-get_max_prompt_length(state) :]
        prompt = torch.tensor(prompt).unsqueeze(0).to(models_device, dtype=torch.long)

        generator = self._generator(
            idx=prompt,
            max_new_tokens=state.get("max_new_tokens", 10),
            temperature=state.get("temperature", 1.0),
            top_k=state.get("top_k", 1),
        )
        output = ""
        for token in generator:
            if shared.stop_everything:
                break

            if callback:
                callback(token)

            output += token

        return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ""
            for token in generator:
                reply += token
                yield reply

    @torch.no_grad()
    def _generator(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete the
        sequence max_new_tokens times, feeding the predictions back into the model each time.

        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.model.gpt.model_config.block_size
                else idx[:, -self.model.gpt.model_config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self.model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # hack for llama tokenizer which needs multiple tokens to place word boundaries
            orig = self.decode(idx.flatten().tolist())
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            yield self.decode(idx.flatten().tolist())[len(orig) :]
