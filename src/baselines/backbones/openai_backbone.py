import os
from typing import Dict, Any, Optional, List

import backoff
import openai
from dotenv import load_dotenv
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.utils.type_utils import ChatMessage
from src.baselines.prompts.base_prompt import BugLocalizationPrompt
from src.baselines.prompts.file_list_prompt import FileListPrompt
from src.baselines.prompts.utils import batch_project_context
from src.utils.git_utils import get_repo_content_on_commit
from src.utils.hf_utils import load_data


class OpenAIBackbone(BaseBackbone):

    def __init__(
            self,
            model_name: str,
            prompt: BugLocalizationPrompt,
            parameters: Dict[str, Any],
            api_key: Optional[str] = None,
    ):
        self._client = openai.OpenAI(api_key=api_key if api_key else os.environ.get("OPENAI_API_KEY"))
        self._model_name = model_name
        self._prompt = prompt
        self._parameters = parameters

    @staticmethod
    def name():
        return 'openai'

    @backoff.on_exception(backoff.expo, openai.APIError)
    def _get_chat_completion(self, messages: List[ChatMessage]) -> ChatCompletion:
        return self._client.chat.completions.create(messages=messages, model=self._model_name, **self._parameters)  # type: ignore[arg-type]

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str]) -> Dict[str, Any]:
        batched_project_contents = batch_project_context(
            self._model_name, self._prompt, issue_description, repo_content, True
        )

        expected_files = set()
        raw_completions = []
        for batched_project_content in batched_project_contents:
            messages = self._prompt.chat(issue_description, batched_project_content)

            completion = self._get_chat_completion(messages)
            raw_completions.append(completion)
            for file in completion.choices[0].message.content.split('\n'):

                if file in repo_content:
                    expected_files.add(file)

        return {
            "expected_files": list(expected_files),
            "raw_completions": raw_completions
        }


def main():
    load_dotenv()
    baseline = OpenAIBackbone('gpt-4-0613', FileListPrompt(), {})
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
