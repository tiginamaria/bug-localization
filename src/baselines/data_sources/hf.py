import os
from typing import List, Optional

from datasets import get_dataset_config_names, load_dataset  # type: ignore[import-untyped]

from .base import BaseDataSource
from ...utils.git_utils import get_repo_content_on_commit


class HFDataSource(BaseDataSource):

    def __init__(
            self,
            hub_name: str,
            configs: Optional[List[str]] = None,
            split: Optional[str] = None,
            cache_dir: Optional[str] = None,
    ):
        self._hub_name = hub_name
        self._cache_dir = cache_dir

        if configs:
            self._configs = configs
        else:
            self._configs = get_dataset_config_names(self._hub_name)
        self._split = split

    def _load_repos(self):
        # TODO: Copy paste repos loading to here
        pass

    def __iter__(self):
        for config in self._configs:
            dataset = load_dataset(self._hub_name, config, split=self._split, cache_dir=self._cache_dir)
            self._load_repos()
            for dp in dataset:
                repo_path = os.path.join(self._cache_dir, f"{dp['repo_owner']}__{dp['repo_name']}")
                extensions = [config] if config != 'mixed' else None
                # Move parameters to data source config
                repo_content = get_repo_content_on_commit(repo_path, dp['base_sha'],
                                                          extensions=extensions,
                                                          ignore_tests=True)
                yield dp, repo_content
