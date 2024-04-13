from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.configs.backbone_configs import BackboneConfig, OpenAIBackboneConfig
from src.configs.data_configs import DataSourceConfig, HFDataSourceConfig
from src.configs.prompt_configs import FileListPromptConfig


@dataclass
class BaselineConfig:
    backbone: BackboneConfig = MISSING
    data_src: DataSourceConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="baseline_config", node=BaselineConfig)
# all available options for the backbone
cs.store(name="openai", group="backbone", node=OpenAIBackboneConfig)
# all available options for the prompt
cs.store(name="simple", group="backbone/prompt", node=FileListPromptConfig)
# all available options for the input
cs.store(name="hf", group="data_src", node=HFDataSourceConfig)
