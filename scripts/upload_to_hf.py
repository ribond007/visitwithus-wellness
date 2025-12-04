# scripts/upload_to_hf.py
import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file, create_repo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_USERNAME = os.getenv("HF_USERNAME", "RishiBond")
DATASET_REPO = f"{HF_USERNAME}/visitwithus-wellness-data"
MODEL_REPO = f"{HF_USERNAME}/visitwithus-wellness-model-docker"

api = HfApi(token=HF_TOKEN)

def ensure_repo(repo_id, repo_type="dataset"):
    try:
        create_repo(repo_id.split("/")[-1], repo_type=repo_type, token=HF_TOKEN, exist_ok=True)
        print(f"Repo {repo_id} ensured.")
    except Exception as e:
        print("Repo create/ensure may have issue:", e)

def upload_files():
    # dataset
    ds_file = DATA_DIR / "tourism.csv"
    if ds_file.exists():
        ensure_repo(DATASET_REPO, repo_type="dataset")
        upload_file(path_or_fileobj=str(ds_file), path_in_repo="tourism.csv", repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
        print("Uploaded dataset to HF:", DATASET_REPO)
    else:
        print("Dataset not found:", ds_file)

    # train/test splits if exist
    for split in ["train.csv", "test.csv", "cleaned.csv"]:
        p = DATA_DIR / split
        if p.exists():
            upload_file(path_or_fileobj=str(p), path_in_repo=split, repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
            print("Uploaded", split)

    # model
    model_file = MODEL_DIR / "best_model.pkl"
    if model_file.exists():
        ensure_repo(MODEL_REPO, repo_type="model")
        upload_file(path_or_fileobj=str(model_file), path_in_repo="best_model.pkl", repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN)
        # upload a simple README
        readme = MODEL_DIR / "README.md"
        readme.write_text("# VisitWithUs - Best Model\nModel pickled with scikit-learn pipeline.\n")
        upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN)
        print("Uploaded model to HF:", MODEL_REPO)
    else:
        print("Model file not found:", model_file)

if __name__ == "__main__":
    upload_files()
