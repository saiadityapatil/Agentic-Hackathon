import httpx
import re

GITHUB_API = "https://api.github.com"
MAX_FILE_SIZE = 200_000  # 200 KB per file limit
ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".tf", ".json"}

# CI/CD files to extract (by exact name or pattern)
CICD_FILES = {
    "Dockerfile", "dockerfile",
    "Jenkinsfile", "jenkinsfile",
    ".gitlab-ci.yml", ".gitignore",
    ".github/workflows", "github/workflows",
    "azure-pipelines.yml", "azure-pipelines.yaml",
    ".circleci/config.yml",
    ".travis.yml",
    "buildspec.yml",
    "cloudbuild.yaml",
    "Makefile", "makefile",
    ".dockerignore",
    "docker-compose.yml", "docker-compose.yaml"
}

def parse_github_url(repo_url: str):
    """
    Extract owner and repo from GitHub URL
    """
    pattern = r"github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, repo_url)
    if not match:
        raise ValueError("Invalid GitHub URL")
    owner = match.group(1)
    repo = match.group(2).replace(".git", "")
    return owner, repo


def fetch_repo_contents(owner: str, repo: str, path: str = ""):
    """
    Fetch contents of a repo path
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    with httpx.Client() as client:
        response =  client.get(url)
        response.raise_for_status()
        return response.json()


def is_allowed_file(file_name: str, file_path: str = "") -> bool:
    """
    Check if file should be extracted based on extension or CI/CD naming
    """
    # Check standard extensions
    if any(file_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return True
    
    # Check CI/CD files by name
    if file_name in CICD_FILES:
        return True
    
    # Check for CI/CD files in specific paths
    cicd_patterns = [
        ".github/workflows/",
        ".gitlab/",
        ".circleci/",
        ".github/",
    ]
    
    full_path = file_path or file_name
    if any(pattern in full_path for pattern in cicd_patterns):
        return True
    
    return False


def extract_repo_code(repo_url: str):
    """
    Main function:
    Recursively extract allowed files from public GitHub repo
    """
    owner, repo = parse_github_url(repo_url)
    collected_files = {}

    def recursive_fetch(path=""):
        items =  fetch_repo_contents(owner, repo, path)

        if isinstance(items, dict) and items.get("type") == "file":
            # Single file
             process_file(items)
             return

        for item in items:
            if item["type"] == "dir":
                 recursive_fetch(item["path"])

            elif item["type"] == "file":
                 process_file(item)

    def process_file(file_item):
        file_name = file_item["name"]
        file_path = file_item.get("path", "")

        # Filter by extension or CI/CD naming
        if not is_allowed_file(file_name, file_path):
            return

        # Skip large files
        if file_item.get("size", 0) > MAX_FILE_SIZE:
            return

        # Download raw file
        with httpx.Client() as client:
            response =  client.get(file_item["download_url"])
            response.raise_for_status()
            collected_files[file_item["path"]] = response.text

    recursive_fetch()

    return collected_files