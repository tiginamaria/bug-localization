import hydra
from dotenv import load_dotenv

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.configs.baseline_configs import BaselineConfig
from src.baselines.data_sources.base import BaseDataSource


@hydra.main(version_base="1.1", config_path="../../configs/baselines")
def main(cfg: BaselineConfig) -> None:
    backbone: BaseBackbone = hydra.utils.instantiate(cfg.backbone)
    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_src)

    results_path = cfg.output_path
    for dp, repo_content in data_src:
        result = backbone.localize_bugs(dp['issue_text'], repo_content)
        result['text_id'] = dp['text_id']


if __name__ == '__main__':
    load_dotenv()
    main()
