from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dataset import MetaMathQA
import re
import os
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from pprint import pprint
from datasets import Dataset, DatasetDict

# SPECIAL AND EXTRA SPECIAL TOKEN

SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "mask_token": "[MASK]",
}
EXTRA_SPECIAL_TOKENS = ["[SYS]", "[USR]", "[AST]", "[/SYS]", "[/USR]", "[/AST]"]

# System Prompt
SYSTEM_PROMPT_VI = (
    'Bạn là một trợ lý toán học.'
    'Hãy giải bài toán chi tiết và rõ ràng bằng tiếng Việt.'
)
SYSTEM_PROMPT_EN = (
    'You are an expert in Mathematics.'
    'Solve this problem clearly.'
)

VOCAB_SIZE    = 32000
MIN_FREQUENCY = 2
MAX_SEQ_LEN   = 1024
BPE_SAVE_PATH = "bpe_tokenizer"
PROCESSED_PATH = "processed_dataset"


def clean_text(text):
    if not text:
        return ''

    text = text.strip('')
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Công thức nằm trong $ $ - Cú pháp của Latex 'Khoảng cách giữa hai điểm $(x_1,y_1)$ và $(x_2,y_2)$ trong '
    parts = re.split(r"(\$[^$]*\$|\$\$[^$]*\$\$)", text)
    cleaned = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            part = re.sub(r" {2,}", " ", part)
        cleaned.append(part)
    return "".join(cleaned)

'''
Chat Text for Training (casual LM)
[SYS] SYSTEM PROMPT [/SYS]
[USR] USER QUERY [/USR]
[AST] CHATBOT RESPONSE [/AST]
[EOS]
'''

def build_chat_text_vi(query, response):
    chat_text = (
        f'[SYS]{SYSTEM_PROMPT_VI}[/SYS]'
        f'[USR]{query}[/USR]'
        f'[AST]{response}[/AST]'
        f'[EOS]'
    )
    return chat_text

def build_chat_text_en(query, response):
    chat_text = (
        f'[SYS]{SYSTEM_PROMPT_EN}[/SYS]'
        f'[USR]{query}[/USR]'
        f'[AST]{response}[/AST]'
        f'[EOS]'
    )
    return chat_text


def iter_training_texts(data_path='dataset'):
    meta = MetaMathQA(data_path=data_path)
    # len(dataset) = 395000
    total = meta.__len__() # 395000

    for i in tqdm(range(total), desc='Dataset preprocessing'):
        row = meta.get_item_vi(i)
        query = clean_text(row.get("query_vi") or '')
        response = clean_text(row.get("response_vi") or '')

        if len(query) > 2 and len(response) > 2:
            yield build_chat_text_vi(query, response)


# BPE (Byte Pair Encoding)
def bpe_tokenizer(data_path='dataset', save_path=BPE_SAVE_PATH, vocab_size=VOCAB_SIZE):
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.normalizer = NFKC()
    tokenizer.decoder = ByteLevelDecoder()

    all_special = list(SPECIAL_TOKENS.values()) + EXTRA_SPECIAL_TOKENS

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=all_special,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()
    )

    print('Start BPE!!!')
    tokenizer.train_from_iterator(iter_training_texts(data_path), trainer=trainer)

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(os.path.join(save_path, 'tokenizer.json'))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=MAX_SEQ_LEN,
        **SPECIAL_TOKENS,
    )
    fast_tokenizer.add_special_tokens({'additional_special_tokens': EXTRA_SPECIAL_TOKENS})
    fast_tokenizer.save_pretrained(save_path)

    print('BPE tokenizer trained. Vocab size: {}'.format(fast_tokenizer.vocab_size))
    print('save to: {}'.format(save_path))
    return fast_tokenizer

def load_bpe_tokenizer(save_path=BPE_SAVE_PATH):
    print('load BPE tokenizer!!')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(save_path)
    print('vocab size: {}'.format(tokenizer.vocab_size))
    return tokenizer

def _tokenize(tokenizer, data_path='dataset', save_path=PROCESSED_PATH, max_length=MAX_SEQ_LEN):
    meta = MetaMathQA(data_path=data_path)
    total = meta.__len__()

    all_input_ids = []
    all_labels = []
    all_attn_mask = []

    skipped = 0

    for i in tqdm(range(total), desc='Tokenizing!!!'):
        row = meta.get_item_vi(i)
        query = clean_text(row.get("query_vi") or '')
        response = clean_text(row.get("response_vi") or '')

        if len(query) <= 2 or len(response) <= 2:
            skipped += 1
            continue

        text = build_chat_text_vi(query, response)

        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        # {ids: , attention_mask:}
        ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        all_input_ids.append(ids)
        all_labels.append(ids.copy())
        all_attn_mask.append(attn_mask)

    print("Total: {}. Accepted: {}. Skipped: {}".format(total, len(all_input_ids), skipped))

    # Token Length
    lengths = [len(x) for x in all_input_ids]
    print('Token length - AVG: {} | max: {}, min: {}'.format(
        sum(lengths) / len(lengths), max(lengths), min(lengths))
    )

    # Save split tokenized dataset
    full_ds = Dataset.from_dict({
        'input_ids': all_input_ids,
        'labels': all_labels,
        'attention_mask': all_attn_mask,
    })

    split = full_ds.train_test_split(test_size=0.2, seed=42)
    ds_dict = DatasetDict({
        'train': split['train'],
        'eval': split['test'],
    })
    print('Train: {}. Eval: {}'.format(len(ds_dict['train']), len(ds_dict['eval'])))
    os.makedirs(save_path, exist_ok=True)
    ds_dict.save_to_disk(save_path)
    print('saved to: {}'.format(save_path))

    return ds_dict





if __name__ == '__main__':
    dataset = MetaMathQA('processed_dataset/train')
    # vi_prob = dataset.get_item_vi(0)
    # print(clean_text(vi_prob['response_vi']))
    # tokenizer = load_bpe_tokenizer()
    # _tokenize(tokenizer)
    print(dataset.get_item_vi(1))