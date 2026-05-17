import os
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
repo_id = "Axeeel/imdb-sentiment-nn"
api = HfApi()

api.create_repo(repo_id, token=token, exist_ok=True)

for filename in ["sentiment_model.pth", "vectorizer.pkl", "model_info.json"]:
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=repo_id,
        token=token,
    )
    print(f"Uploaded {filename}")

print(f"Done! https://huggingface.co/{repo_id}")
