import multiprocessing
import os

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.utils.hf_utils import CATEGORIES


def add_stats(dp, dp_info):
    dp['repo_symbols_count'] = dp_info['repo_symbols_count']
    dp['repo_tokens_count'] = dp_info['repo_tokens_count']
    dp['repo_lines_count'] = dp_info['repo_lines_count']
    dp['repo_files_without_tests_count'] = dp_info['repo_files_without_tests_count']

    dp['changed_symbol_count'] = dp_info['changed_symbol_count']
    dp['changed_tokens_count'] = dp_info['changed_tokens_count']
    dp['changed_lines_count'] = dp_info['changed_lines_count']
    dp['changed_files_without_tests_count'] = dp_info['changed_files_without_tests_count']

    dp['issue_links_count'] = dp_info['issue_links_count']
    dp['issue_code_blocks_count'] = dp_info['issue_code_blocks_count']

    return dp


def prepare_dataset(config: DictConfig):
    stats_df = pd.read_csv(os.path.join(config.bug_localization_data_path, 'metrics.csv'))

    for category in CATEGORIES:
        df = pd.read_csv(os.path.join(config.bug_localization_data_path, f"bug_localization_data_{category}.csv"))

        params = [(dp, stats_df.loc[stats_df['text_id'] == dp["text_id"]].squeeze()) for _, dp in df.iterrows()]

        cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=cpus) as pool:
            results = pool.starmap(add_stats, params)

        results = [dp for dps in results for dp in dps]

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(config.bug_localization_data_path, f"bug_localization_data_{category}.csv"),
                  escapechar="\\", index=False)
        df.to_json(os.path.join(config.bug_localization_data_path, f"bug_localization_data_{category}.jsonl"),
                   orient="records", lines=True)


@hydra.main(config_path="../../../configs/data", config_name="server", version_base=None)
def main(config: DictConfig):
    prepare_dataset(config)


if __name__ == "__main__":
    load_dotenv()
    main()
