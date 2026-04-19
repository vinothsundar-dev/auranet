import zipfile, os

PROJECT = "/Users/vinoth-14902/auranet"
zip_path = os.path.expanduser("~/Desktop/auranet_project.zip")

include_files = [
    "model.py", "model_optimized.py", "model_optimized_v2.py",
    "model_v3.py", "loss.py", "loss_v3.py", "loss_perceptual.py",
    "dataset.py", "dataset_v3.py",
    "train.py", "train_v2.py", "train_v2_stable.py", "train_v3.py",
    "train_finetune.py", "metrics.py",
    "infer.py", "infer_v3.py",
    "export.py", "export_v3.py",
    "config.yaml", "config_v3.yaml",
    "requirements.txt",
    "postprocessing.py", "profiling.py", "quantization.py", "validation.py",
    "quick_diagnostic.py",
    "debug_training.py", "debug_loss_balance.py",  # Debug scripts
    "auranet_v2_complete.py", "auranet_v2_edge.py", "auranet_v2_optimized.py",
    "utils/__init__.py", "utils/audio_utils.py", "utils/stft.py",
    "deploy/benchmark.py", "deploy/convert_coreml.py",
    "deploy/convert_tflite.py", "deploy/export_onnx.py",
    "scripts/download_dns_subset.py",
    "docs/QUICK_REFERENCE.md", "docs/VSCODE_COLAB_SETUP.md",
    "notebooks/AuraNet_Kaggle_Training.ipynb",
    "notebooks/AuraNet_Dev_Template.ipynb",
    "AuraNet_Training_Colab.ipynb",
]

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in include_files:
        full = os.path.join(PROJECT, f)
        if os.path.exists(full):
            zf.write(full, f)
            print("  + " + f)
        else:
            print("  SKIP " + f)

size_kb = os.path.getsize(zip_path) / 1024
print("\nCreated: " + zip_path + " (" + str(int(size_kb)) + " KB)")
