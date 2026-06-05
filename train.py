import warnings
import torch
import os
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="timm"
)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from datetime import datetime
import json

from argparse import ArgumentParser

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    set_seed,
    BitsAndBytesConfig,
    pipeline, AutoModelForMaskedLM,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer, SFTConfig
import bitsandbytes as bnb
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser

from preprocessing import (
    load_bpe_tokenizer,
    BPE_SAVE_PATH,
    PROCESSED_PATH,
    MAX_SEQ_LEN,
    SYSTEM_PROMPT_VI,
    clean_text
)
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)


# LORA MODULES
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Prompt management for training and inference

class MathPromptBuilder:
    TRAIN_TEMPLATE = (
        '[SYS]{system}[/SYS]'
        '[USR]{query}[/USR]'
        '[AST]{response}[/AST]'
        '[EOS]'
    )
    INFERENCE_TEMPLATE = (
        '[SYS]{system}[/SYS]'
        '[USR]{query}[/USR]'
        '[AST]'
    )
    def __init__(self, system_prompt=SYSTEM_PROMPT_VI):
        self.system_prompt = system_prompt

        self.train_prompt = PromptTemplate(
            input_variables=['system', 'query', 'response'],
            template=self.TRAIN_TEMPLATE,
        )
        self.inference_prompt = PromptTemplate(
            input_variables=['system', 'query'],
            template=self.INFERENCE_TEMPLATE,
        )

    def build_training_text(self, query, response):
        return self.train_prompt.format(
            system=self.system_prompt,
            query=clean_text(query),
            response=clean_text(response)
        )
    def build_inference_prompt(self, query):
        return self.inference_prompt.format(
            system=self.system_prompt,
            query=clean_text(query),
        )

    def validate(self, text):
        required = ["[SYS]", "[/SYS]", "[USR]", "[/USR]", "[AST]"]
        return all(tag in text for tag in required)

# Model Wrapper: HuggingFacePipeline -> LangChain LLM
class InferenceChain:
    def __init__(self, model, tokenizer, prompt_builder: MathPromptBuilder):
        hf_pipe = pipeline(
            task='text-generation',
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)
        self.prompt_builder = prompt_builder
        self.parser = StrOutputParser()

    def run(self, query):
        prompt = self.prompt_builder.build_inference_prompt(query)
        raw_output = self.llm.invoke(prompt)
        answer = self.parser.invoke(raw_output)
        return answer

    def run_batch(self, queries):
        results = []
        for query in queries:
            results.append(self.run(query))
        return results

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

def get_lora_config(args):
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )

def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable : {trainable:,} / {total:,}  ({100 * trainable / total:.3f}%)")

def load_datasets(args):
    ds = load_from_disk(args.processed_path)
    train_ds = ds['train']
    eval_ds = ds['eval']

    if args.max_samples:
        n = min(args.max_samples, len(train_ds))
        train_ds = train_ds.select(range(n))
        eval_ds = eval_ds.select(range(max(1, n // 20)))
        print(f"Giới hạn : train={len(train_ds):,} | eval={len(eval_ds):,}")
    else:
        print(f"Train : {len(train_ds):,} | Eval : {len(eval_ds):,}")

    return train_ds, eval_ds

# Logger
class LoggerCallback(TrainerCallback):
    def __init__(self, log_dir='log'):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, 'training_log.jsonl')
        self.parser = StrOutputParser()

    def _enrich(self, entry):
        entry['timestamp'] = datetime.now().strftime('%H:%M:%S')
        if 'loss' in entry:
            entry['loss'] = round(entry['loss'], 4)
        if 'eval_loss' in entry:
            entry['eval_loss'] = round(entry['eval_loss'], 4)
        if 'learning_rate' in entry:
            entry['learning_rate'] = round(entry['learning_rate'], 4)

        return entry

    def _write(self, entry):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _to_message(self, entry):
        parts = [f"step={entry['step']:>6}", f"epoch={entry['epoch']}"]
        if "loss" in entry:
            parts.append(f"loss={entry['loss']}")
        if "eval_loss" in entry:
            parts.append(f"eval_loss={entry['eval_loss']}")
        if "learning_rate" in entry:
            parts.append(f"lr={entry['learning_rate']}")

        raw_message = "📊 " + " | ".join(parts)
        return self.parser.invoke(raw_message)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 3), **logs}
        entry = self._enrich(entry)
        self._write(entry)
        print(self._to_message(entry))

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"📝 Log → {self.log_path}")

    def on_train_end(self, args, state, control, **kwargs):
        best = state.best_metric
        msg = f"best eval_loss={best:.4f}" if best else "hoàn tất"
        print(f"🏁 Training {msg}")

def argument():
    parser = ArgumentParser(description='LLM Fine-Tuning')

    parser.add_argument('--base_model', type=str, default='sail/Sailor-4B-Chat')

    parser.add_argument('--bpe_path', type=str, default=BPE_SAVE_PATH)
    parser.add_argument('--processed_path', type=str, default=PROCESSED_PATH)
    parser.add_argument('--output_dir', type=str, default='outputs')

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    # parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    return args

def train():
    args = argument()
    set_seed(args.seed)

    prompt_builder = MathPromptBuilder()
    tokenizer = load_bpe_tokenizer(args.bpe_path)
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Trainable : 778,426,880 / 2,364,398,080  (32.923%)
    if len(tokenizer) > model.config.vocab_size:
        print(f"Resizing embeddings: {model.config.vocab_size:,} to {len(tokenizer):,}")
        model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print('QLoRA - r={} | alpha={} | dropout={}'.format(args.lora_r, args.lora_alpha, args.lora_dropout))
    model = get_peft_model(model, get_lora_config(args))

    print("Trainable Params")
    print_trainable_params(model)

    train_ds, eval_ds = load_datasets(args)

    log_dir = os.path.join(args.output_dir, 'logs')
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_32bit",
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        logging_dir=log_dir,
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        remove_unused_columns=False,
        seed=args.seed,
        max_length=args.max_seq_len,
        packing=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=100,
        pad_to_multiple_of=8,
        padding=True
    )

    callback = LoggerCallback(log_dir=log_dir)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[callback],
    )


if __name__ == '__main__':
    train()
