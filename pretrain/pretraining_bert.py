import torch
from torch.utils.data import Dataset
from transformers import (
                            AutoConfig,
                            AutoModelForMaskedLM,
                            BertConfig, 
                            BertForMaskedLM,
                            BertTokenizerFast, 
                            BertForPreTraining,
                            DataCollatorForLanguageModeling, 
                            LineByLineTextDataset,
                            Trainer, 
                            TrainingArguments, 
                            PreTrainedTokenizer
                        )
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple
from torch.nn.utils.rnn import pad_sequence
import linecache
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding


# memory size:
# use global data: data size x 2
# not use global data: data size x 3

print(f"torch is available: {torch.cuda.is_available()}")

config = BertConfig(
    vocab_size=32_000,
    attention_probs_dropout_prob=0.1,
    directionality="bidi",
    gradient_checkpointing=False,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    layer_norm_eps=1e-12,
    max_position_embeddings=512,
    model_type="bert",
    num_attention_heads=12,
    num_hidden_layers=12,
    pad_token_id=0,
    pooler_fc_size=768,
    pooler_num_attention_heads=12,
    pooler_num_fc_layers=3,
    pooler_size_per_head=128,
    pooler_type="first_token_transform",
    type_vocab_size=2,
)
tokenizer = BertTokenizerFast.from_pretrained("../bertwordpiece_32000", max_len=32)

model = BertForMaskedLM(config=config)

# config = AutoConfig.from_pretrained('bert-base-multilingual-cased')
# print(config)
# model = AutoModelForMaskedLM.from_pretrained('bert-base-multilingual-cased', 
                                            # config=config)

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = BertForPreTraining.from_pretrained("bert/checkpoint-1470000").to(device)

print(f"model parameters: {model.num_parameters()}")
# => 68 million parameters


class LazyLineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, dir_path: str, block_size: int, data_size: int):
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        textlines = []
        file_list = Path(dir_path).glob('*.txt')
        for file_path in tqdm(file_list, desc='read_corpus'):
            with open(file_path, encoding='utf-8') as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
                textlines += lines
                if len(textlines) > data_size:
                    break

        self.textlines = textlines

        print(f"total size of corpus: {len(self.textlines)}")

    def __len__(self):
        return len(self.textlines)

    def __getitem__(self, i) -> str:
        return self.textlines[i]


def _collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


@dataclass
class LazyDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    block_size: int = 32

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def _tokenizer(self, lines):
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)
        examples = batch_encoding["input_ids"]
        examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in examples]
        return examples

    def __call__(self, lines: List[str]) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        examples = self._tokenizer(lines)
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


train_dataset = LazyLineByLineTextDataset(
    dir_path='../../data/naver_news/train',
    block_size=32,
    data_size=300_000_000
)

eval_dataset = LazyLineByLineTextDataset(
    dir_path='../../data/naver_news/eval',
    block_size=32,
    data_size=30_000_000
)

# train_dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path='../data/naver_news/news_2_kss/train/naver_news00.txt',
#     block_size=32,
# )

# eval_dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path='../data/naver_news/news_2_kss/eval/naver_news25.txt',
#     block_size=32,
# )

data_collator = LazyDataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, block_size=32
)

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
# )

training_args = TrainingArguments(
    output_dir="./bert",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=384,
    per_device_eval_batch_size=384,
    save_steps=1_000,
    save_total_limit=2,
    logging_dir='./logs',
    max_grad_norm = 1.0,
    prediction_loss_only=True,
    do_train=True, 
    do_eval=True,
    dataloader_num_workers=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
