from datasets import load_dataset
def download_datasets():
    datasets_info = [
        {"name": "cnn_dailymail", "config": "3.0.0", "trust_remote_code": False},
        {"name": "xsum", "config": None, "trust_remote_code": True},
        {"name": "multi_news", "config": None, "trust_remote_code": True},
        {"name": "gigaword", "config": None, "trust_remote_code": True},
        {"name": "big_patent", "config": "a", "trust_remote_code": True},
        {"name": "samsum", "config": None, "trust_remote_code": True},
    ]

    save_dir = "./artifacts/data_ingestion"
    for info in datasets_info:
        dataset = load_dataset(info["name"], info["config"], trust_remote_code=info["trust_remote_code"])
        dataset.save_to_disk(f"{save_dir}/{info["name"]}")

if __name__ == "__main__":
    download_datasets()