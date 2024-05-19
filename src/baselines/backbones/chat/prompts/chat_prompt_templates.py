FILE_LIST_PROMPT_TEMPLATE = """
    List of files in project: {}
    Bug issue: {}
    You are given a list of files in a project and a bug issue description.
    Select a subset of 1-5 files which SHOULD be fixed according to the issue.
    Provide output in JSON format with one field 'files' with a list of file names which SHOULD be fixed.
    Make sure that all file names in the answer appear in the project files list.
    Sort files from most probable to change to less probable.
    Provide ONLY json without any additional comments.
"""
