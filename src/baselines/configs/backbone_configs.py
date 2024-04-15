from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from omegaconf import MISSING

from src.baselines.configs.prompt_configs import PromptConfig


@dataclass
class BackboneConfig:
    _target_: str = MISSING
    prompt: Optional[PromptConfig] = None
    model_name: str = MISSING


@dataclass
class OpenAIBackboneConfig(BackboneConfig):
    _target_: str = f"src.baselines.backbones.OpenAIBackbone"
    api_key: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=lambda: {})
