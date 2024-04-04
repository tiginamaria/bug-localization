import datasets
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.utils.hf_utils import update_hf_data_splits


def split_data(df: datasets.Dataset, split: str, test_data_ids: list[str]):
    test_data_ids = [i.lower() for i in test_data_ids]
    if split == 'dev':
        return df

    if split == 'test':
        return df.filter(lambda dp: dp['text_id'] in test_data_ids)

    if split == 'train':
        return df.filter(lambda dp: dp['text_id'] not in test_data_ids)


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def run_split_data(config: DictConfig):
    update_hf_data_splits(
        lambda df, category, split: split_data(df, split, config.test_data_ids),
    )


if __name__ == '__main__':
    load_dotenv()
    run_split_data()
