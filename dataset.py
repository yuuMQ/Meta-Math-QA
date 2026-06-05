from dotenv import load_dotenv
import os
from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from pprint import pprint

# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")
# login(hf_token, add_to_git_credential=True)

class MetaMathQA:
    def __init__(self, data_path='dataset'):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            self._download_dataset()

        self.dataset = load_from_disk(self.data_path)

    def _download_dataset(self):
        dataset = load_dataset('5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated', split='train')
        dataset.save_to_disk(self.data_path)
        os.makedirs(self.data_path, exist_ok=True)
        dataset.save_to_disk(self.data_path)

    def __len__(self):
        return len(self.dataset)

    def get_item_vi(self, index):
        row = self.dataset[index]
        target = ['original_question_vi', 'query_vi', 'response_vi', 'type']
        result = {k: row[k] for k in target}
        return result

    def get_item_en(self, index):
        row = self.dataset[index]
        target = ['original_question_en', 'query_en', 'response_en', 'type']
        result = {k: row[k] for k in target}
        return result

if __name__ == '__main__':
    dataset = MetaMathQA()
    pprint(dataset.get_item_vi(0))
    '''
    Dataset format:
    original_question (both en and vi)
    query (en and vi)
    response (en and vi)
    type
    '''


