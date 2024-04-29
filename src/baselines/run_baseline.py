import csv
import os

import hydra
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.configs.baseline_configs import BaselineConfig
from src.baselines.data_sources.base import BaseDataSource


@hydra.main(version_base="1.1", config_path="../../configs/baselines")
def main(cfg: BaselineConfig) -> None:
    backbone: BaseBackbone = hydra.utils.instantiate(cfg.backbone)
    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_src)

    results_path = os.path.join(cfg.output_path, cfg.name)
    os.makedirs(results_path, exist_ok=True)

    OmegaConf.save(config=cfg, f=f"{cfg.name}.yaml")
    results_csv_path = os.path.join(results_path, "results.csv")

    for dp, repo_content, changed_files in data_src:
        issue_description = dp['issue_title'] + '\n' + dp['issue_body']
        results_dict = backbone.localize_bugs(issue_description, repo_content)
        results_dict['text_id'] = dp['text_id']

        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(results_dict.keys())
            writer.writerow(results_dict.values())


if __name__ == '__main__':
    load_dotenv()
    main()
