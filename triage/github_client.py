from github import Github

from . import config


class Repo:
    def __init__(self):
        self.repo = Github(config.GITHUB_TOKEN).get_repo(config.GITHUB_REPOSITORY)

    def get_issue(self, number: int):
        return self.repo.get_issue(number)

    def other_issues(self, exclude: int, limit: int = 100) -> list[dict]:
        out = []
        for i in self.repo.get_issues(state="all"):
            if i.pull_request or i.number == exclude:
                continue
            out.append({"number": i.number, "title": i.title, "body": i.body})
            if len(out) >= limit:
                break
        return out

    def read_file(self, path: str, max_chars: int = 4000) -> str:
        return self.repo.get_contents(path).decoded_content.decode()[:max_chars]
