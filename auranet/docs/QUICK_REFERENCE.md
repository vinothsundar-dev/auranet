# VS Code + Colab Quick Reference

## 🚀 Quick Start

### One-time Setup

```bash
# VS Code: Install extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter

# Local: Create venv
cd ~/auranet
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Daily Workflow

```
VS Code                          Colab
───────                          ─────
1. Edit code                     
2. Test: python train.py --debug 
3. git add -A                    
4. git commit -m "msg"           
5. git push                ────► 6. !git pull
                                 7. Run training
                                 8. Checkpoints → Drive
```

---

## 📋 Colab Setup Cell (Copy-Paste)

```python
# === COLAB SETUP (Run First) ===
import os, sys

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone/update repo
REPO = "https://github.com/YOUR_USERNAME/auranet.git"
if not os.path.exists('/content/auranet'):
    !git clone {REPO} /content/auranet
else:
    !cd /content/auranet && git pull

%cd /content/auranet

# Install deps
!pip install -q torch torchaudio librosa numpy soundfile tqdm pyyaml

# Set paths
CHECKPOINT_DIR = "/content/drive/MyDrive/AuraNet/checkpoints"
DATASET_DIR = "/content/drive/MyDrive/AuraNet/datasets"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 🔧 Common Commands

### Git (VS Code Terminal)
```bash
git pull                          # Get latest
git add -A && git commit -m "x"   # Stage & commit
git push                          # Push to GitHub
git status                        # Check status
```

### Colab Git
```python
!git pull                                    # Update
!git add -A && git commit -m "x" && git push # Push back
```

### Training
```bash
# Local (debug)
python train.py --debug --epochs 1

# Colab (full)
!python train.py --epochs 100 --checkpoint-dir /content/drive/MyDrive/AuraNet/checkpoints
```

---

## 📁 Path Reference

| What | Local | Colab |
|------|-------|-------|
| Project | `~/auranet/` | `/content/auranet/` |
| Drive | N/A | `/content/drive/MyDrive/` |
| Checkpoints | `./checkpoints/` | `/content/drive/MyDrive/AuraNet/checkpoints/` |
| Datasets | `./datasets/` | `/content/drive/MyDrive/AuraNet/datasets/` |

---

## ⚠️ Remember

- ✓ Enable GPU: Runtime → Change runtime type → GPU
- ✓ Mount Drive before accessing files
- ✓ Save checkpoints to Drive (not /content/)
- ✓ Pull latest code before training
- ✗ /content/ clears on disconnect!
