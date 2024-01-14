import together
# import wandb

# wandb.login(key='9ab5cdd0bc2e1863667fdb6a11525504204e6635')

# Set your API key for Together
together.api_key = "74ffcb9226b29abfd2611d40ed3dbdd62309437789c5e3d6cc1de348e90a5d77"  # Replace with your actual API key

import together
import json

# Define file paths and model
file_path = "./concatenated_output.jsonl"
transformed_file_path = "./transformed_concatenated_output.jsonl"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

def transform_string(input_str):
    # Apply the transformations based on the specified rules
    transformed_str = input_str
    transformed_str = transformed_str.replace("<s>[SYSTEM]", "<s>[INST] <<SYS>>")
    transformed_str = transformed_str.replace("[USER]", "<<\\SYS>>")
    transformed_str = transformed_str.replace("[ASSISTANT]", "[/INST]")
    transformed_str = transformed_str.replace("</s>", "</s>")

    return transformed_str

def transform_file(input_file_path, output_file_path):
    """Transforms the entire file content."""
    with open(input_file_path, 'r') as file, open(output_file_path, 'w') as outfile:
        for line in file:
            data = json.loads(line)
            transformed_text = transform_string(data["text"])
            json.dump({"text": transformed_text}, outfile)
            outfile.write("\n")

def check_file_format(file_path):
    """Check the format of the JSONL file."""
    resp = together.Files.check(file=file_path)
    print("Format check response:", resp)
    return resp.get('is_check_passed', False)

def upload_file(file_path):
    """Uploads the file if the format is correct."""
    resp = together.Files.upload(file=file_path)
    print("Upload response:", resp)
    return resp.get('id') if resp['report_dict'].get('is_check_passed') else None

def start_finetuning(file_id, model_name):
    """Starts a fine-tuning job with the given file ID and model."""
    response = together.Finetune.create(
        training_file=file_id,
        model=model_name,
        n_epochs=3,
        n_checkpoints=1,
        batch_size=4,
        learning_rate=1e-5,
        suffix="finetune-example",
        wandb_api_key="9ab5cdd0bc2e1863667fdb6a11525504204e6635",
    )
    return response

transform_file(file_path, transformed_file_path)
# Check the file format
if check_file_format(file_path):
    # Upload the file and start fine-tuning if the format is correct
    file_id = upload_file(transformed_file_path)
    if file_id:
        print(f"File uploaded successfully. File ID: {file_id}")
        # Start fine-tuning
        response = start_finetuning(file_id, model_name)
        print(response)
    else:
        print("Failed to upload the file.")
else:
    print("File format check failed.")