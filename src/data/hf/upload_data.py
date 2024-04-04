import os

import huggingface_hub
import hydra
from datasets import DatasetDict, Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.utils.hf_utils import CATEGORIES, HUGGINGFACE_REPO


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def upload_bug_localization_data(config: DictConfig):
    huggingface_hub.login(token=os.environ['HUGGINGFACE_TOKEN'])

    for category in CATEGORIES:
        df = Dataset.from_json(
            os.path.join(config.bug_localization_data_path, f'bug_localization_data_{category}.jsonl'),
        )
        dataset_dict = DatasetDict({'dev': df})
        dataset_dict.push_to_hub(HUGGINGFACE_REPO, category)


if __name__ == '__main__':
    load_dotenv()
    upload_bug_localization_data()
