from pprint import pprint
from datasets import load_from_disk
from dataset import MetaMathQA
from preprocessing import build_chat_text_vi, clean_text, load_bpe_tokenizer

class InspectData:
    def __init__(self, data_path='dataset', processed_path='processed_dataset', bpe_path='bpe_tokenizer', n=3, start=0, decode=True):
        self.data_path = data_path
        self.n = n
        self.start = start
        self.meta = MetaMathQA(self.data_path)
        self.processed_path = processed_path
        self.bpe_path = bpe_path
        self.decode = decode

    def inspect_raw(self):
        print('Dataset gốc từ vị trí {} đến {}:'.format(self.start, self.start + self.n - 1))
        for i in range(self.start, self.start + self.n):
            row = self.meta.get_item_vi(i)
            pprint(row)

    def inspect_chat_format(self):
        print('Chat format từ vị trí {} đến {}:'.format(self.start, self.start + self.n - 1))
        for i in range(self.start, self.start + self.n):
            row = self.meta.get_item_vi(i)
            query = clean_text(row.get('query_vi') or '')
            response = clean_text(row.get('response_vi') or '')
            text = build_chat_text_vi(query, response)
            pprint(text)

    def _indent(self, text, prefix='  '):
        return '\n'.join(prefix + line for line in text.splitlines()) or prefix

    def inspect_tokenized(self):
        ds_dict = load_from_disk(self.processed_path)

        train_ds = ds_dict['train']
        eval_ds = ds_dict['eval']

        print('Train split: {}'.format(len(train_ds)))
        print('Eval split: {}'.format(len(eval_ds)))

        tokenizer = None
        if self.decode:
            tokenizer = load_bpe_tokenizer(self.bpe_path)

        print('Sample từ vị trí {} đến {}:'.format(self.start, self.start + self.n - 1))
        for i in range(self.start, self.start + self.n):
            sample = train_ds[i]
            ids = sample['input_ids']
            attn_mask = sample['attention_mask']
            labels = sample['labels']

            print('input ids length: {}'.format(len(ids)))
            print('input_ids (đầu): {}'.format(ids[:20]))
            print('input_ids (cuối): {}'.format(ids[-10:]))
            print('labels == input: {}'.format(ids == labels))
            print('attention mask: {}'.format(attn_mask[:20]))

            if self.decode and tokenizer:
                decoded = tokenizer.decode(ids, skip_special_tokens=False)
                print('Decoded (600 ký tự đầu):')
                print(self._indent(decoded[:600] + ('...' if len(decoded) > 600 else ''), '│    '))

    def inspect_vocab(self, top_n=30):
        tokenizer = load_bpe_tokenizer(self.bpe_path)

        vocab = tokenizer.get_vocab()
        print('Vocab size: {}'.format(len(vocab)))

        print('Special tokens:')
        for name, token in [
            ('PAD', tokenizer.pad_token),
            ('UNK', tokenizer.unk_token),
            ('BOS', tokenizer.bos_token),
            ('EOS', tokenizer.eos_token),
        ]:
            tid = tokenizer.convert_tokens_to_ids(token)
            print(f"{name:6} → '{token}' (id={tid})")

        extra = tokenizer.all_special_tokens
        print('extra special tokens: {}'.format(extra))

        print('Test Encode/Decode:')
        tests = [
            "Tính diện tích hình tròn bán kính 5 cm.",
            "Giải phương trình $x^2 - 5x + 6 = 0$",
            "The answer is $\\frac{{3}}{{4}}$.",
        ]
        for t in tests:
            ids = tokenizer.encode(t)
            tokens = tokenizer.convert_ids_to_tokens(ids)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            print(f"\n  Input  : {t}")
            print(f"  Tokens : {tokens[:12]}{'...' if len(tokens) > 12 else ''}")
            print(f"  Len    : {len(ids)}")
            print(f"  Decoded: {decoded}")


if __name__ == '__main__':
    inspect_data = InspectData()
    # inspect_data.inspect_raw()
    # inspect_data.inspect_chat_format()
    # inspect_data.inspect_tokenized()
    inspect_data.inspect_vocab()