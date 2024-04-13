from typing import List, Dict

from src.baselines.prompts.base_prompt import BugLocalizationPrompt
from src.utils.tokenization_utils import TokenizationUtils


def check_match_context_size(tokenization_utils: TokenizationUtils,
                             prompt: BugLocalizationPrompt,
                             issue_description: str,
                             project_content: Dict[str, str],
                             is_chat: bool):
    if is_chat:
        messages = prompt.chat(issue_description, project_content)
        return tokenization_utils.messages_match_context_size(messages)

    text = prompt.complete(issue_description, project_content)
    return tokenization_utils.text_match_context_size(text)


def batch_project_context(model: str,
                          prompt: BugLocalizationPrompt,
                          issue_description: str,
                          project_content: Dict[str, str],
                          is_chat: bool) -> List[Dict[str, str]]:
    tokenization_utils = TokenizationUtils(model)
    file_paths = list(project_content.keys())

    has_big_message = True
    n = len(file_paths)
    step = len(file_paths)

    while has_big_message:
        has_big_message = False
        for i in range(0, n, step):
            project_content_subset = {f: c for f, c in project_content.items() if f in file_paths[i:i + step]}
            if not check_match_context_size(tokenization_utils, prompt, issue_description, project_content_subset,
                                            is_chat):
                has_big_message = True
                step //= 2
                break

    batched_project_content = [
        {f: c for f, c in project_content.items() if f in file_paths[i:i + step]} for i in range(0, n, step)
    ]
    assert len(file_paths) == sum(len(b) for b in batched_project_content)

    return batched_project_content
