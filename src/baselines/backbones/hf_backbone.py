import os
from typing import Dict, Any

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.prompts.file_list_prompt import FileListPrompt
from src.baselines.prompts.utils import batch_project_context
from src.utils.git_utils import get_repo_content_on_commit
from src.utils.hf_utils import load_data


class HfBackbone(BaseBackbone):

    def __init__(self, model: str):
        self.model = model

    @staticmethod
    def name():
        return 'hf'

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str]) -> Dict[str, Any]:
        prompt = FileListPrompt()
        batched_project_contents = batch_project_context(self.model, prompt, issue_description, repo_content, True)

        expected_files = set()
        for batched_project_content in batched_project_contents:
            messages = prompt.chat(issue_description, batched_project_content)

            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=True,
                                                         torch_dtype=torch.bfloat16).cuda()
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
                model.device)
            outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95,
                                     num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print(output)

            for file in output.split('\n'):
                if file in repo_content:
                    expected_files.add(file)

        return {
            "expected_files": list(expected_files)
        }


def main():
    load_dotenv()
    baseline = HfBackbone('deepseek-ai/deepseek-coder-1.3b-instruct')
    config = OmegaConf.load("../../../configs/data/server.yaml")

    df = load_data('java', 'test')
    for dp in df:
        repo_path = os.path.join(config.repos_path, f"{dp['repo_owner']}__{dp['repo_name']}")
        repo_content = get_repo_content_on_commit(repo_path, dp['base_sha'], ['java'])
        result = baseline.localize_bugs(
            dp['issue_body'],
            repo_content,
        )
        print(result)
        print(dp['changed_files'])

        return


if __name__ == '__main__':
    main()
