from typing import Dict, Any

import numpy as np
import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.backbones.emb.rankers.base_ranker import BaseRanker
from src.baselines.utils.embed_utils import data_to_vectors


class HfEmbBackbone(BaseBackbone):

    def __init__(self,
                 pretrained_path: str,
                 model_name: str,
                 parameters: Dict[str, Any],
                 ranker: BaseRanker):
        self._pretrained_path = pretrained_path
        self._model_name = model_name
        self._parameters = parameters
        self._ranker = ranker

    name: str = "huggingface"

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str]) -> Dict[str, Any]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self._model_name, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).cuda()

        batch_size = self._parameters.get("batch_size", 1)
        file_names, file_contents = data_to_vectors(issue_description, repo_content)
        vect_file_contents = []

        for i in range(0, len(file_contents), batch_size):
            inputs = tokenizer(file_contents[i: (i + batch_size)],
                               return_tensors="pt",
                               padding='max_length',
                               truncation=True,
                               return_attention_mask=False).to(model.device)
            batch_embeddings = model(**inputs)
            batch_embeddings = batch_embeddings.to('cpu')

            vect_file_contents.append(batch_embeddings.detach().numpy())
            del inputs
            del batch_embeddings

        vect_file_contents = np.concatenate(vect_file_contents)
        assert len(file_contents) == vect_file_contents.shape[0]

        ranked_file_names, rank_scores = self._ranker.rank(file_names, vect_file_contents)

        return {
            "file_names": ranked_file_names,
            "rank_scores": rank_scores
        }
