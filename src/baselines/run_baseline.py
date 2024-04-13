import hydra

from src.configs.baseline_configs import BaselineConfig


@hydra.main(version_base="1.1", config_path="configs", config_name="baseline_config")
def main(cfg: BaselineConfig) -> None:
