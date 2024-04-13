from src.baselines.prompts.base_prompt import BugLocalizationPrompt


class FileListPrompt(BugLocalizationPrompt):

    def _base_prompt(self, issue_description: str, project_content: dict[str, str]) -> str:
        file_paths = '\n'.join(project_content.keys())

        return f"""
            List of files:"
            {file_paths}
            Issue:
            {issue_description}
            You are given a list of files in project and bug issue description.
            Select subset of 1-20 files which SHOULD be fixed according to issue.
            Provide output in JSON format with one field 'files' with list of files which SHOULD be fixed.
            Provide ONLY json without any additional comments.
        """
