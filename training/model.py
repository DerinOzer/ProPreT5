import os
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration, BertTokenizer

import idr_torch  # provides .rank, .local_rank, .size


def buildTokenizer(config):
    """
    Build a BERT-style tokenizer from a plain-text vocab file.
    Creates a local 'vocab.txt' that starts with special tokens, then the custom tokens
    (and their '##' wordpiece variants)
    """
    vocab_file = config.vocab_path

    # Rank 0 creates vocab.txt; other ranks wait.
    if (not config.multigpu) or (idr_torch.rank == 0):
        with open(vocab_file, "r", encoding="utf-8") as f:
            # Strip blanks/whitespace-only lines
            vocab = [line.strip() for line in f if line.strip()]

        with open("vocab.txt", "w", encoding="utf-8") as f:
            # Special tokens first (one per line)
            f.write("<pad>\n")
            f.write("<unk>\n")
            f.write("<bos>\n")
            f.write("<eos>\n")
            f.write("<cls>\n")
            f.write("<sep>\n")
            f.write("<mask>\n")

            # Then the custom tokens and their '##' prefixed forms
            for idx, token in enumerate(vocab):
                f.write(token + "\n")
                f.write("##" + token)
                if idx != len(vocab) - 1:
                    f.write("\n")

    if config.multigpu and idr_torch.rank > 0:
        print(f"{idr_torch.rank}_Waiting for main process to build the tokenizer")
        dist.barrier()

    tokenizer = BertTokenizer(
        vocab_file="vocab.txt",
        do_lower_case=False,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
    )

    # Ensure special tokens are registered (no-op if already present)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "bos_token": "<bos>",
            "unk_token": "<unk>",
            "cls_token": "<cls>",
            "sep_token": "<sep>",
            "mask_token": "<mask>",
        }
    )

    print("vocab size :", tokenizer.vocab_size)

    if config.multigpu and idr_torch.rank == 0:
        print(f"{idr_torch.rank}_Unlock building tokenizer")
        dist.barrier()

    return tokenizer


class ProPreT5(nn.Module):
    """
    Thin wrapper around T5 to use custom token/type embeddings.
    - If `is_code_generator` is False or `is_prior` is True -> full encoder-decoder (T5ForConditionalGeneration)
    - Else -> encoder-only (T5EncoderModel)
    """

    def __init__(self, config, tokenizer, is_code_generator: bool = False, is_prior: bool = False):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.config_t5 = T5Config(
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
            unk_token_id=self.tokenizer.unk_token_id,
        )
        print(
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.sep_token_id,
        )

        if (not is_code_generator) or is_prior:
            self.model = T5ForConditionalGeneration(config=self.config_t5)
        else:
            self.model = T5EncoderModel(config=self.config_t5)

        self.input_embeddings = nn.Embedding(
            self.tokenizer.vocab_size,
            self.model.model_dim,
            padding_idx=self.tokenizer.pad_token_id,
        )
        self.type_embeddings = nn.Embedding(len(self.config.types), self.model.model_dim)
        print("pad_idx =", self.tokenizer.pad_token_id)

    def forward(self, input_ids, attention_mask, types, target_ids):
        """
        Args:
            input_ids: [batch, seq]
            attention_mask: [batch, seq]
            types: [batch, seq] integer type IDs aligned with input_ids
            target_ids: labels for seq2seq LM (T5)
        """
        inputs_embeds = self.input_embeddings(input_ids) + self.type_embeddings(types)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=target_ids,
            output_hidden_states=True,
        )
        return outputs

    def generate(self, input_ids, attention_mask, types, **kwargs):
        """
        Generation wrapper that builds inputs_embeds and a 1-token decoder start.
        Respects config.max_len and config.num_beams by default.
        """
        inputs_embeds = self.input_embeddings(input_ids) + self.type_embeddings(types)

        # Start decoder with pad token (as configured)
        decoder_input_ids = torch.ones(
            (inputs_embeds.shape[0], 1), dtype=torch.long, device=input_ids.device
        ) * self.model.config.decoder_start_token_id

        output_ids = self.model.generate(
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            inputs_embeds=inputs_embeds,
            max_length=self.config.max_len,
            num_beams=self.config.num_beams,
            **kwargs,
        )
        return output_ids
