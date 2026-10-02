import argparse
from datasets import load_dataset
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Push SAHA-AL Benchmark to Hugging Face")
    parser.add_argument("--repo_id", type=str, default="huggingbahl21/saha-al", help="Hugging Face Hub Repo ID")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train/validation/test jsonl files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Token (optional if already logged in via huggingface-cli)")
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.data_dir}...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{args.data_dir}/train.jsonl",
            "validation": f"{args.data_dir}/validation.jsonl",
            "test": f"{args.data_dir}/test.jsonl"
        }
    )
    
    print(f"Pushing dataset to Hugging Face hub ({args.repo_id})...")
    dataset.push_to_hub(args.repo_id, token=args.token)
    
    print("Uploading README.md and evaluation script...")
    api = HfApi(token=args.token)
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset"
    )
    api.upload_file(
        path_or_fileobj="benchmark_eval.py",
        path_in_repo="benchmark_eval.py",
        repo_id=args.repo_id,
        repo_type="dataset"
    )
    
    print("Success! Your benchmark is now live.")

if __name__ == "__main__":
    main()
