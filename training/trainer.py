import os
import csv
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import idr_torch
from datasets import Dataset, Features, Value, concatenate_datasets, load_from_disk
from datasets.utils.logging import set_verbosity_info
from sacrebleu import BLEU
from tqdm import tqdm
from transformers.trainer_pt_utils import LengthGroupedSampler, DistributedLengthGroupedSampler

from dataCollator import DataCollatorForSeq2SeqCustom

set_verbosity_info()

logger = logging.getLogger(__name__)


def split_on(s: str, char: str = ".") -> Tuple[str, str]:
    """Split once on the first occurrence of `char`."""
    parts = s.split(char, 1)
    return (parts[0], parts[1]) if len(parts) > 1 else (s, "")


def print_number_of_trainable_model_parameters(model: torch.nn.Module) -> str:
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        n = param.numel()
        all_model_params += n
        if param.requires_grad:
            trainable_model_params += n
    pct = 100 * trainable_model_params / all_model_params if all_model_params else 0.0
    return (
        f"trainable model parameters: {trainable_model_params}\n"
        f"all model parameters: {all_model_params}\n"
        f"percentage of trainable model parameters: {pct:.2f}%"
    )


def get_norm_of_model_parameters(model: torch.nn.Module) -> torch.Tensor:
    """Return L2 norm of all parameters (as a scalar tensor)."""
    accum = torch.tensor(0.0, device=next(model.parameters()).device)
    for _, param in model.named_parameters():
        accum += torch.norm(param, p=2) ** 2
    return torch.sqrt(accum)

def get_rxn_list():
    reaclist = []
    # Append Atom (Add CNOF to CNO)
    reaclist.append('[#6,#7,#8;h:1].[O,N,F,C:2]>>[#6,#7,#8:1][O,N,F,C:2]')

    # bonds
    # X-X -> X=X
    reaclist.append('[O,N,C;h:1][O,N,C;h:2]>>[O,N,C:1]=[O,N,C:2]')
    # X-X -> X#X
    reaclist.append('[N,C;h2:1][N,C;h2:2]>>[N,C:1]#[N,C:2]')
    # X=X -> X#X
    reaclist.append('[C;h:1]=[N,C;h:2]>>[C:1]#[N,C:2]')

    # rings
    # close ring 3
    reaclist.append('[#6,#7,#8;h:1]~[*:2]~[#6,#7,#8;h:3]>>[#6,#7,#8:1]1[*:2]~[#6,#7,#8:3]1')
    # close ring 4
    reaclist.append('[#6,#7,#8;h:1]~[*:2]~[*:4]~[#6,#7,#8;h:3]>>[#6,#7,#8:1]1[*:2]~[*:4]~[#6,#7,#8:3]1')
    # close ring 5
    reaclist.append('[#6,#7,#8;h:1]~[*:2]~[*:4]~[*:5]~[#6,#7,#8;h:3]>>[O,N,C:1]1[*:2]~[*:4]~[*:5]~[#6,#7,#8:3]1')
    # close ring 6
    reaclist.append('[#6,#7,#8;h:1]~[*:2]~[*:4]~[*:5]~[*:6]~[#6,#7,#8;h:3]>>[O,N,C:1]1[*:2]~[*:4]~[*:5]~[*:6]~[#6,#7,#8:3]1')
    # close ring 7
    reaclist.append('[#6,#7,#8;h:1]~[*:2]~[*:4]~[*:5]~[*:6]~[*:7]~[#6,#7,#8;h:3]>>[O,N,C:1]1[*:2]~[*:4]~[*:5]~[*:6]~[*:7]~[#6,#7,#8:3]1')
    # close ring 8
    reaclist.append('[#6,#7,#8;h:1]~[*:2]~[*:4]~[*:5]~[*:6]~[*:7]~[*:8]~[#6,#7,#8;h:3]>>[O,N,C:1]1[*:2]~[*:4]~[*:5]~[*:6]~[*:7]~[*:8]~[#6,#7,#8:3]1')
    return reaclist


class Trainer:
    def __init__(self, config, device: torch.device, tokenizer, writer: SummaryWriter):
        self.config = config
        self.writer = writer
        self.output_dir = config.output_dir
        self.device = device
        self.tokenizer = tokenizer
        self.reactions = get_rxn_list()

    def _preprocess_function_custom(self, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # Collect keys and number of examples
        keys = list(examples.keys())
        nb_ex = len(examples[keys[0]]) if keys else 0

        # Expand inputs/targets to include numbered variants (_1, _2, ...)
        def sorted_like(elem_names: List[str]) -> List[str]:
            return [
                key
                for element in elem_names
                for key in sorted(
                    (k for k in keys if k == element or k.startswith(f"{element}_")),
                    key=lambda k: int(k.split("_")[1]) if "_" in k else 0,
                )
            ]

        input_keys = sorted_like(self.config.inputs)
        target_keys = sorted_like(self.config.targets)

        # If reactions are integer indices, replace them with string patterns
        dataset_synthesis = any(
            k.startswith("reaction") and isinstance(v[0], int) for k, v in examples.items()
        )
        if dataset_synthesis:
            examples = {
                k: [self.reactions[r] for r in v] if k.split("_")[0] == "reaction" else v
                for k, v in examples.items()
            }

        # Tokenize all fields (robust to None entries)
        features = {
            k: self.tokenizer(
                [x if x is not None else "" for x in v],
                max_length=self.config.max_len,
                truncation=True,
            )
            for k, v in examples.items()
        }

        # Type IDs aligned to each field
        type_map = {
            k: self.config.types[k] if k in self.config.types else self.config.types[k.split("_")[0]]
            for k in features.keys()
        }
        types_per_key = {k: [[type_map[k]] * len(ids) for ids in v.input_ids] for k, v in features.items()}

        # Flatten across input fields in the configured order
        types = [[t for k in input_keys for t in types_per_key[k][i]] for i in range(nb_ex)]
        ids = [[t for k in input_keys for t in features[k].input_ids[i]] for i in range(nb_ex)]
        mask = [[t for k in input_keys for t in features[k].attention_mask[i]] for i in range(nb_ex)]
        all_ids = [[t for k in features.keys() for t in features[k].input_ids[i]] for i in range(nb_ex)]

        # Build labels (targets + <sep> → replace trailing <sep> with <eos>)
        labels = [
            [t for k in target_keys for t in features[k].input_ids[i] + [self.tokenizer.sep_token_id]]
            for i in range(nb_ex)
        ]
        labels = [l[:-1] + [self.tokenizer.eos_token_id] for l in labels]

        ret = {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels,
            "types": types,
            "all_ids": all_ids,
            "input_lengths": [len(x) for x in ids],
        }
        return ret

    def load_and_prepare_dataset_old(self, dataset_name: str, fraction: float = 1, rebuildCache: bool = True):
        if self.config.multigpu and idr_torch.rank > 0:
            print(f"{idr_torch.rank}_Waiting for main process to perform the mapping")
            dist.barrier()

        cache_dir = os.path.join(os.getenv("HF_DATASETS_CACHE", "~"), f"{dataset_name}_fraction={fraction}")
        print(idr_torch.rank, " load data from csv")

        features = Features(
            {
                "reactant_1": Value("string"),
                "reactant_2": Value("string"),
                "product": Value("string"),
                "reaction": Value("string"),
                "reaction_idx": Value("int32"),
            }
        )

        dataset = Dataset.from_csv(self.config.data_dir + dataset_name, cache_dir=cache_dir, features=features)
        print(idr_torch.rank, " dataset loaded")

        if idr_torch.rank == 0:
            print("special tokens :", self.tokenizer.special_tokens_map)
            print(dataset.features)

        if fraction < 1:
            dataset = dataset.train_test_split(test_size=1 - fraction)["train"]

        load_from_cache = not rebuildCache if idr_torch.rank == 0 else True
        print(idr_torch.rank, " len dataset=", dataset_name, ":", len(dataset), "loadFromCache ", load_from_cache)

        remove_columns = sorted(dataset.features.keys())
        cache_name = os.path.join(cache_dir, "preprocessed_data.arrow")

        print(idr_torch.rank, " cache : before map ", dataset.cache_files)
        dataset = dataset.map(
            self._preprocess_function_custom,
            batched=True,
            remove_columns=remove_columns,
            load_from_cache_file=load_from_cache,
            cache_file_name=cache_name,
            batch_size=1000,
        )
        print(idr_torch.rank, " cache : after map ", dataset.cache_files)

        if self.config.multigpu and idr_torch.rank == 0:
            print("Dataset processed ! ", len(dataset))
            print(dataset.features)
            a = next(iter(dataset))
            print(a["input_ids"])
            print(a["attention_mask"])
            print(a["types"])
            print(a["labels"])
            print(self.tokenizer.decode(a["labels"], skip_special_tokens=True))
            print("Unlock from master")
            dist.barrier()

        if self.config.multigpu:
            print(idr_torch.rank, " barrier ! wait others")
            dist.barrier()

        return dataset

    def _build_dataset_from_master(self, dataset_names: List[str], fraction, fullname: str, rebuildCache: bool):
        print(idr_torch.rank, " build dataset ", fullname, " from ", dataset_names)
        datasets = []

        for i, ds_name in enumerate(dataset_names):
            frac = fraction[i] if isinstance(fraction, list) else fraction
            print(idr_torch.rank, " load data ", ds_name, " from csv")

            name = f"{ds_name}_fraction={frac}"
            cache_dir = os.path.join(os.getenv("HF_DATASETS_CACHE", "~"), name)
            os.makedirs(cache_dir, exist_ok=True)

            dataset = Dataset.from_csv(self.config.data_dir + ds_name, cache_dir=cache_dir)
            print(dataset.features)

            if frac < 1:
                dataset = dataset.train_test_split(test_size=1 - frac)["train"]

            remove_columns = sorted(dataset.features.keys())
            cache_name = os.path.join(cache_dir, "preprocessed_data.arrow")

            dataset = dataset.map(
                self._preprocess_function_custom,
                batched=True,
                remove_columns=remove_columns,
                load_from_cache_file=(not rebuildCache),
                cache_file_name=cache_name,
                batch_size=1000,
            )
            datasets.append(dataset)
            print("Dataset ", name, " processed ! ", len(dataset))
            print(dataset.features)
            a = next(iter(dataset))
            print(a["input_ids"])
            print(a["attention_mask"])
            print(a["types"])
            print(a["labels"])
            print(self.tokenizer.decode(a["labels"], skip_special_tokens=True))

        print("Consolidate ", str(len(datasets)), " datasets")
        dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
        dataset.save_to_disk(os.path.join(os.getenv("HF_DATASETS_CACHE", "~"), f"{fullname}_preprocessed"))
        return dataset

    def load_and_prepare_dataset(self, dataset_names, fraction=1, rebuildCache: bool = True):
        if not isinstance(dataset_names, list):
            dataset_names = [dataset_names]

        if self.config.multigpu and idr_torch.rank > 0:
            print(f"{idr_torch.rank}_Waiting for main process to perform the mapping")
            dist.barrier()

        # Build a unique cache key
        parts = []
        for i, ds_name in enumerate(dataset_names):
            frac = fraction[i] if isinstance(fraction, list) else fraction
            parts.append(f"{ds_name}_fraction={frac}")
        fullname = "+".join(parts)

        cache_dir = os.path.join(os.getenv("HF_DATASETS_CACHE", "~"), f"{fullname}_preprocessed")

        if idr_torch.rank == 0:
            print("load ", cache_dir, " if exists")
            if (not os.path.isdir(cache_dir)) or rebuildCache:
                dataset = self._build_dataset_from_master(dataset_names, fraction, fullname, rebuildCache)
            else:
                dataset = load_from_disk(cache_dir)
                print(idr_torch.rank, " dataset ", fullname, " loaded from cache")
                print(dataset.features)

            if self.config.multigpu:
                print("Unlock from master")
                dist.barrier()
        else:
            print(idr_torch.rank, " load data ")
            dataset = load_from_disk(cache_dir)
            print(idr_torch.rank, " dataset ", fullname, " loaded")

        if self.config.multigpu:
            print(idr_torch.rank, " barrier ! wait others")
            dist.barrier()

        return dataset

    def collate(self, data_collator, features, return_tensors=None):
        return data_collator(features, return_tensors)

    def getDataLoader(self, dataset: Dataset, tokenizer, batchsize: int) -> DataLoader:
        data_collator = DataCollatorForSeq2SeqCustom(
            tokenizer=tokenizer,
            label_pad_token_id=tokenizer.pad_token_id,
            extra_pads=["input_ids", "attention_mask", "types", "labels", "all_ids"],
        )

        if self.config.multigpu:
            sampler = DistributedLengthGroupedSampler(
                batch_size=batchsize,
                dataset=dataset,
                num_replicas=idr_torch.size,
                rank=idr_torch.rank,
                lengths=dataset["input_lengths"],
            )
        else:
            sampler = LengthGroupedSampler(
                batch_size=batchsize,
                dataset=dataset,
                lengths=dataset["input_lengths"],
            )

        print("Sampler initialized")
        return DataLoader(
            dataset,
            batchsize,
            sampler=sampler,
            collate_fn=data_collator,
            pin_memory=True,
            shuffle=False,
            num_workers=0,
        )

    def run_training(self, model, dataset_train: Dataset, device: torch.device, dataset_test: Dataset = None):
        batch_size_per_gpu = self.config.batch_size
        print("create train dataloader")
        dataloader_train = self.getDataLoader(dataset_train, self.tokenizer, batch_size_per_gpu)

        dataloader_test = None
        if dataset_test is not None:
            print("create test dataloader")
            dataloader_test = self.getDataLoader(dataset_test, self.tokenizer, batch_size_per_gpu)

        print("dataloader created ! ", len(dataloader_train))

        if self.config.multigpu:
            model = DDP(model, device_ids=[idr_torch.local_rank])

        optimizer = AdamW(model.parameters(), lr=self.config.lr, weight_decay=0.0)

        if self.config.start_from is not None:
            self.load_model(self.config.start_from, self.config.start_epoch, model, optimizer)

        print(print_number_of_trainable_model_parameters(model))

        step = 0
        rank_prefix = f"{idr_torch.rank}_" if self.config.multigpu else ""

        for epoch in range(self.config.start_epoch + 1, self.config.start_epoch + 1 + self.config.epochs):
            model.train()
            total_loss = torch.zeros((1), device=self.device)
            total_predictions = torch.zeros((1), device=self.device)
            total_correct = torch.zeros((1), device=self.device)

            batch_iterator = tqdm(dataloader_train, desc=f"Processing train epoch {epoch:02d}")
            for batch in batch_iterator:
                step += 1
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                types = batch["types"].to(device, non_blocking=True)
                target_ids = batch["labels"].to(device, non_blocking=True)

                # Apply training masks per type
                for k, v in self.config.training_masked_types.items():
                    psample, mask_rate = v
                    masked_samples = torch.bernoulli(torch.ones((input_ids.shape[0], 1), device=self.device) * psample)
                    masked_tokens = torch.bernoulli(
                        torch.ones_like(input_ids, dtype=torch.float32, device=self.device) * masked_samples * mask_rate
                    )
                    masked_tokens = (masked_tokens == 1) & (types == k)
                    input_ids = torch.where(masked_tokens, torch.full_like(input_ids, self.tokenizer.pad_token_id), input_ids)
                    attention_mask = torch.where(masked_tokens, torch.zeros_like(attention_mask), attention_mask)

                # Remap types if requested
                for k, v in self.config.remap_types.items():
                    types = torch.where(types == k, torch.ones_like(input_ids) * v, types)

                # Ignore pads in loss
                target_ids[target_ids == self.tokenizer.pad_token_id] = -100

                # Forward / loss
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, types=types, target_ids=target_ids
                )
                loss = outputs.loss

                predicted_token_ids = torch.argmax(outputs.logits, dim=-1)

                if (not self.config.multigpu) or idr_torch.rank == 0:
                    norm = get_norm_of_model_parameters(model)
                    if step % 100 == 0:
                        print(f"{rank_prefix}step{step}_norm_params= {norm}")
                    self.writer.add_scalar("norm model", norm, step)

                    if step % 100 == 0:
                        print(f"{rank_prefix}step{step}_target_ids", target_ids[0])
                        print(f"{rank_prefix}step{step}_predicted_token_ids", predicted_token_ids[0])

                with torch.no_grad():
                    correct = ((target_ids == predicted_token_ids) * (target_ids >= 0)).sum()
                    nb = (target_ids > 0).sum()

                    total_correct += correct
                    total_predictions += nb
                    total_loss += loss * nb

                    if step % 100 == 0:
                        print(f"{rank_prefix}step{step}_correct_rate= {correct / nb}")

                    suffix = f"{idr_torch.rank}" if self.config.multigpu else ""
                    self.writer.add_scalar(f"accuracy/accuracy_{suffix}", correct.item() / nb.item(), step)
                    self.writer.add_scalar(f"loss/loss_{suffix}", loss.item(), step)

                if step % 100 == 0:
                    print(f"{rank_prefix}step{step}_train loss = {loss.item()}")

                batch_iterator.set_postfix({f"train loss": f"{loss.item():6.3f}"})

                loss.backward()
                optimizer.step()

            # Sync metrics across ranks
            if self.config.multigpu:
                dist.barrier()
                dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_predictions, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                print("all reduce ok")
                dist.barrier()

            avg_loss = total_loss.item() / total_predictions.item()
            accuracy = total_correct.item() / total_predictions.item()

            if (not self.config.multigpu) or idr_torch.rank == 0:
                self.writer.add_scalar("accuracy/avg_accuracy", accuracy, epoch)
                self.writer.add_scalar("loss/avg_loss", avg_loss, epoch)

                module = model.module if self.config.multigpu else model
                self.save_model(self.config.save_dir, epoch, module, optimizer, None)

            if dataloader_test is not None:
                self.run_validation(model, dataloader_test, device, batch_size_per_gpu, epoch)

    def run_validation(self, model, dataloader_test: DataLoader, device: torch.device, batch_size_per_gpu: int, epoch: int):
        generations_file = os.path.join(self.config.output_dir, f"generations_epoch{epoch}.csv")

        model.eval()
        test_loss = torch.tensor(0.0, device=device)
        total_correct = torch.zeros((1), device=self.device)
        total_predictions = torch.zeros((1), device=self.device)

        step = 0
        gensteps = 0
        bleu = BLEU()
        bleu_score = torch.zeros((1), device=self.device)

        rank_prefix = f"{idr_torch.rank}_" if self.config.multigpu else ""

        with torch.no_grad():
            batch_iterator = tqdm(dataloader_test, desc=f"Processing test epoch {epoch:02d}")
            for batch in batch_iterator:
                step += 1

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                types = batch["types"].to(device, non_blocking=True)
                target_ids = batch["labels"].to(device, non_blocking=True)
                target_ids[target_ids == self.tokenizer.pad_token_id] = -100
                full_input = batch["all_ids"].to(device, non_blocking=True)

                # CSV header (only once on rank 0)
                if step == 1 and ((not self.config.multigpu) or (idr_torch.rank == 0)):
                    with open(generations_file, mode="w", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow(["step", "rank", "full_input", "input", "target", "generation"])

                # Apply testing masks
                for k, v in self.config.testing_masked_types.items():
                    psample, mask_rate = v
                    masked_samples = torch.bernoulli(torch.ones((input_ids.shape[0], 1), device=self.device) * psample)
                    masked_tokens = torch.bernoulli(
                        torch.ones_like(input_ids, dtype=torch.float32, device=self.device) * masked_samples * mask_rate
                    )
                    masked_tokens = (masked_tokens == 1) & (types == k)
                    input_ids = torch.where(masked_tokens, torch.full_like(input_ids, self.tokenizer.pad_token_id), input_ids)
                    attention_mask = torch.where(masked_tokens, torch.zeros_like(attention_mask), attention_mask)

                # Remap types if requested
                for k, v in self.config.remap_types.items():
                    types = torch.where(types == k, torch.full_like(types, v), types)

                # Forward
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, types=types, target_ids=target_ids
                )
                loss = outputs.loss
                predicted_token_ids = torch.argmax(outputs.logits, dim=-1)

                if (not self.config.multigpu) or idr_torch.rank == 0:
                    if step % 100 == 0:
                        print(f"{rank_prefix}step{step}_target_ids", target_ids[0])
                        print(f"{rank_prefix}step{step}_predicted_token_ids", predicted_token_ids[0])

                correct = ((target_ids == predicted_token_ids) * (target_ids >= 0)).sum()
                nb = (target_ids > 0).sum()

                total_correct += correct
                total_predictions += nb
                test_loss += loss * nb

                if step % 100 == 0:
                    print(f"{rank_prefix}step{step}_test correct_rate= {correct / nb}")
                    print(f"{rank_prefix}step{step}_test loss = {loss.item()}")

                batch_iterator.set_postfix({f"test loss": f"{loss.item():6.3f}"})

                # Periodic generation (kept identical in spirit, but safe on single-GPU)
                if (self.config.freq_gen < 0) or (step % self.config.freq_gen == 0) or (epoch % 10 == 0):
                    gen_model = getattr(model, "module", model)  # safe access for DDP/non-DDP
                    generated_ids = gen_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        types=types,
                        max_length=self.config.max_len,
                        return_dict_in_generate=True,
                    )

                    target_seq = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
                    target_seq = [seq.replace("<unk>", "") for seq in target_seq]
                    print("targets  : ", target_ids[0], target_seq[0])

                    gen_seq = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    print("generations : ", generated_ids[0], gen_seq[0])

                    bleu_score += bleu.corpus_score(gen_seq, [target_seq]).score
                    print("bleu :", bleu_score)
                    gensteps += 1

                    input_seq = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                    input_seq = [seq.replace("<pad>", "") for seq in input_seq]

                    full_input_seq = self.tokenizer.batch_decode(full_input, skip_special_tokens=False)
                    full_input_seq = [seq.replace("<pad>", "") for seq in full_input_seq]

                    # Rank-ordered writing (unchanged behavior)
                    global_rank = idr_torch.rank
                    world_size = dist.get_world_size()
                    for rank in range(world_size):
                        if global_rank == rank:
                            with open(generations_file, mode="a", newline="", encoding="utf-8") as file:
                                writer = csv.writer(file)
                                for i in range(len(generated_ids)):
                                    writer.writerow([step, rank, full_input_seq[i], input_seq[i], target_seq[i], gen_seq[i]])
                        if self.config.multigpu:
                            dist.barrier()

        # Average BLEU over generation steps
        bleu_score = bleu_score / max(gensteps, 1)

        # Sync metrics across ranks
        if self.config.multigpu:
            dist.barrier()
            dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_predictions, op=dist.ReduceOp.SUM)
            dist.all_reduce(bleu_score, op=dist.ReduceOp.AVG)
            dist.barrier()

        avg_loss = test_loss.item() / total_predictions.item()
        accuracy = total_correct.item() / total_predictions.item()

        if (not self.config.multigpu) or idr_torch.rank == 0:
            self.writer.add_scalar("test accuracy/avg_accuracy", accuracy, epoch)
            self.writer.add_scalar("test loss/avg_loss", avg_loss, epoch)
            self.writer.add_scalar("test bleu/avg_bleu", bleu_score, epoch)

        return avg_loss, accuracy

    def recover_test_curves(self, model, dataset_test: Dataset, device: torch.device, epoch_start: int, epoch_end: int):
        batch_size_per_gpu = self.config.batch_size
        print("create test dataloader")
        dataloader_test = self.getDataLoader(dataset_test, self.tokenizer, batch_size_per_gpu)
        print("dataloader created !")

        if self.config.multigpu:
            model = DDP(model, device_ids=[idr_torch.local_rank])

        for epoch in range(epoch_start, epoch_end + 1):
            self.load_model(self.config.start_from, epoch, model)
            self.run_validation(model, dataloader_test, device, batch_size_per_gpu, epoch)

    def save_model(self, save_dir: str, epoch: int, model, optimizer, lr_scheduler=None):
        """Save model/optimizer/scheduler states for a given epoch (1-based naming)."""
        model_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

        optimizer_path = os.path.join(save_dir, f"optimizer_epoch_{epoch + 1}.pth")
        torch.save(optimizer.state_dict(), optimizer_path)

        if lr_scheduler is not None:
            scheduler_path = os.path.join(save_dir, f"scheduler_epoch_{epoch + 1}.pth")
            torch.save(lr_scheduler.state_dict(), scheduler_path)

    def load_model(self, save_dir: str, epoch: int, model, optimizer=None, lr_scheduler=None):
        """
        Load a saved model (and optionally optimizer/scheduler).
        Handles device mapping for multi-GPU vs single-GPU/CPU.
        """
        if self.config.multigpu:
            map_location = {f"cuda:{0}": f"cuda:{idr_torch.local_rank}"}
            rank = idr_torch.rank
        else:
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rank = 0

        model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        checkpoint = torch.load(model_path, map_location=map_location)
        model.module.load_state_dict(checkpoint)
        print(f"Model loaded on GPU {rank}: {model_path}")

        if optimizer is not None:
            optimizer_path = os.path.join(save_dir, f"optimizer_epoch_{epoch}.pth")
            checkpoint = torch.load(optimizer_path, map_location=map_location)
            optimizer.load_state_dict(checkpoint)
            print(f"Optimizer loaded on GPU {rank}: {optimizer_path}")

        if lr_scheduler is not None:
            scheduler_path = os.path.join(save_dir, f"scheduler_epoch_{epoch}.pth")
            checkpoint = torch.load(scheduler_path, map_location=map_location)
            lr_scheduler.load_state_dict(checkpoint)
            print(f"Scheduler loaded on GPU {rank}: {scheduler_path}")

        return model, optimizer, lr_scheduler
