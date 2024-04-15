import hydra
from dotenv import load_dotenv

from src.baselines.configs.baseline_configs import BaselineConfig


@hydra.main(version_base="1.1", config_path="../../configs/baselines")
def main(cfg: BaselineConfig) -> None:
    backbone = hydra.utils.instantiate(cfg.backbone)
    print(backbone._parameters)


if __name__ == '__main__':
    load_dotenv()
    main()
