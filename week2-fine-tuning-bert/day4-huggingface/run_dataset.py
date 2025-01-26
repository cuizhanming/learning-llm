from datasets import load_dataset, load_from_disk

# Load remote dataset from Hugging Face
remote_dataset = load_dataset(
    # data_dir="",
    path="LooksJuicy/ruozhiba",
    # cache_dir="~/.cache/huggingface/datasets",
    cache_dir="../../remote_datasets"
)
print(remote_dataset)


# Load local cached dataset from Hugging Face
local_dataset = load_from_disk(r"../../local_datasets/ChnSentiCorp")
print(local_dataset)

test_data = local_dataset["test"]
print(test_data)
for data in test_data:
    print(data)

# Load csv files from local disk
file_path = r"local_datasets/hermes-function-calling-v1.csv"
local_dataset_csv = load_dataset(path="csv", data_files=file_path)

