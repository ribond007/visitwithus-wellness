from huggingface_hub import HfApi, upload_file
from pathlib import Path
import os

# Space repo id (same as your HF Space)
SPACE_ID = "RishiBond/visitwithus-wellness-app-docker"
HF_TOKEN = os.getenv("HF_TOKEN")  # taken from env / GitHub secrets

ROOT = Path(__file__).resolve().parents[1]

FILES_TO_UPLOAD = [
    ("streamlit_app.py", "streamlit_app.py"),
    ("requirements.txt", "requirements.txt"),
    ("Dockerfile", "Dockerfile"),
    ("README.md", "README.md"),  # optional
]

api = HfApi(token=HF_TOKEN)

def deploy():
    for local_name, repo_path in FILES_TO_UPLOAD:
        local_path = ROOT / local_name
        if local_path.exists():
            print(f"Uploading {local_name} -> {repo_path}")
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=SPACE_ID,
                repo_type="space",
                token=HF_TOKEN,
            )
        else:
            print(f"Skipping {local_name}, file not found.")

if __name__ == "__main__":
    deploy()
