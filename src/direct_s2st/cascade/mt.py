from __future__ import annotations
from ..progress import operation


class NllbTranslator:
    @operation('model: load NLLB')
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        source_language: str,
        target_language: str,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, src_lang=source_language
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, revision=revision
        ).to(self.device)
        self.target_language = target_language

    def __call__(self, text: str) -> str:
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos = self.tokenizer.convert_tokens_to_ids(self.target_language)
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, forced_bos_token_id=forced_bos)
        return str(self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]).strip()
