# 🚀 VS Code + Google Colab Hybrid Workflow Setup

A complete guide for developing AuraNet using VS Code locally and training on Google Colab's free GPU.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [VS Code Setup](#1-vs-code-setup)
3. [Google Colab Setup](#2-google-colab-setup)
4. [Google Drive Integration](#3-google-drive-integration)
5. [GitHub Integration](#4-github-integration)
6. [VS Code ↔ Colab Workflow](#5-vs-code--colab-workflow)
7. [File Sync Strategy](#6-file-sync-strategy)
8. [Running Notebooks in VS Code](#7-running-notebooks-in-vs-code)
9. [Advanced Setup (Optional)](#8-advanced-setup-optional)
10. [Limitations](#9-limitations)
11. [Best Practices](#10-best-practices)

---

## Overview

### Why This Workflow?

| Tool | Best For |
|------|----------|
| **VS Code** | Code editing, IntelliSense, Git, debugging, local testing |
| **Google Colab** | Free GPU (T4/V100), cloud execution, large datasets |
| **Google Drive** | Persistent storage for datasets and checkpoints |
| **GitHub** | Version control, code sync between local and Colab |

### Architecture

```
┌─────────────────┐     Git Push/Pull     ┌─────────────────┐
│                 │ ◄──────────────────► │                 │
│    VS Code      │                       │     GitHub      │
│   (Local Dev)   │                       │   (Code Sync)   │
│                 │                       │                 │
└─────────────────┘                       └────────┬────────┘
                                                   │
                                          Clone/Pull│
                                                   ▼
                                          ┌─────────────────┐
                                          │  Google Colab   │
                                          │  (GPU Training) │
                                          │        │        │
                                          │        ▼        │
                                          │  Google Drive   │
                                          │ (Data/Checkpts) │
                                          └─────────────────┘
```

---

## 1. VS Code Setup

### Step 1.1: Install VS Code

Download from [code.visualstudio.com](https://code.visualstudio.com/)

### Step 1.2: Install Required Extensions

Open VS Code and install these extensions (Cmd+Shift+X / Ctrl+Shift+X):

```
Required:
├── Python (ms-python.python)
├── Jupyter (ms-toolsai.jupyter)
├── Pylance (ms-python.vscode-pylance)
└── GitLens (eamodio.gitlens)

Recommended:
├── GitHub Pull Requests (github.vscode-pull-request-github)
├── YAML (redhat.vscode-yaml)
└── Markdown Preview Enhanced (shd101wyy.markdown-preview-enhanced)
```

**Install via command line:**
```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance
code --install-extension eamodio.gitlens
```

### Step 1.3: Configure Python Interpreter

1. Open Command Palette: `Cmd+Shift+P` (Mac) / `Ctrl+Shift+P` (Windows/Linux)
2. Type: `Python: Select Interpreter`
3. Choose your Python environment (recommend 3.9+)

**Create a virtual environment (recommended):**
```bash
cd ~/auranet
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

pip install torch torchaudio librosa numpy soundfile tqdm pyyaml
```

### Step 1.4: Configure VS Code Settings

Create/edit `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "jupyter.askForKernelRestart": false,
    "notebook.cellToolbarLocation": {
        "default": "right"
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

---

## 2. Google Colab Setup

### Step 2.1: Access Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account

### Step 2.2: Create New Notebook

1. Click **File → New notebook**
2. Or open existing: **File → Open notebook**

### Step 2.3: Enable GPU Runtime

**Critical for training!**

1. Go to **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Choose **T4** (free tier) or **V100/A100** (Colab Pro)
4. Click **Save**

**Verify GPU is available:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Step 2.4: Colab Runtime Limits

| Tier | GPU | Session Limit | Idle Timeout |
|------|-----|---------------|--------------|
| Free | T4 | ~12 hours | ~90 min |
| Pro | T4/V100 | ~24 hours | Longer |
| Pro+ | A100 | ~24 hours | Background |

⚠️ **Important:** Sessions can disconnect. Always save checkpoints to Drive!

---

## 3. Google Drive Integration

### Step 3.1: Mount Drive in Colab

Add this cell at the start of every Colab notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the authorization link and grant access.

### Step 3.2: Set Up Directory Structure in Drive

```
/My Drive/
└── AuraNet/
    ├── datasets/
    │   ├── speech/       # Clean speech audio
    │   ├── music/        # Clean music audio
    │   └── noise/        # Noise samples
    ├── checkpoints/      # Model checkpoints
    ├── outputs/          # Enhanced audio outputs
    └── logs/             # Training logs
```

**Create directories from Colab:**
```python
import os

base_path = "/content/drive/MyDrive/AuraNet"
dirs = ["datasets/speech", "datasets/music", "datasets/noise", 
        "checkpoints", "outputs", "logs"]

for d in dirs:
    os.makedirs(f"{base_path}/{d}", exist_ok=True)
    print(f"✓ Created: {base_path}/{d}")
```

### Step 3.3: Upload Data to Drive

**Option A: Google Drive Web UI**
1. Go to [drive.google.com](https://drive.google.com)
2. Navigate to `AuraNet/datasets/`
3. Drag and drop audio files

**Option B: From Colab (small files)**
```python
from google.colab import files
uploaded = files.upload()  # Opens file picker
```

**Option C: Using gdown (from URL)**
```python
!pip install gdown
!gdown --id YOUR_FILE_ID -O /content/drive/MyDrive/AuraNet/datasets/data.zip
!unzip /content/drive/MyDrive/AuraNet/datasets/data.zip
```

---

## 4. GitHub Integration

### Step 4.1: Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Create repository: `auranet`
3. Initialize with README (optional)

### Step 4.2: Set Up Local Git

```bash
cd ~/auranet

# Initialize git
git init

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/auranet.git

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
venv/
.env

# Data (too large for git)
*.wav
*.mp3
*.flac
datasets/
checkpoints/*.pt

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
EOF

# First commit
git add .
git commit -m "Initial commit: AuraNet project"
git push -u origin main
```

### Step 4.3: Connect Colab to GitHub

**Method A: Open Notebook from GitHub**

1. In Colab: **File → Open notebook**
2. Select **GitHub** tab
3. Enter repository URL or search
4. Select the notebook to open

**Method B: Clone Repo in Colab**

```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/auranet.git /content/auranet

# Navigate to project
%cd /content/auranet

# Pull latest changes
!git pull
```

### Step 4.4: Save Notebook Back to GitHub

**Option A: Save directly to GitHub**

1. **File → Save a copy in GitHub**
2. Select repository and branch
3. Enter commit message
4. Click **OK**

**Option B: Manual commit from Colab**

```python
# Configure git (run once)
!git config --global user.email "your@email.com"
!git config --global user.name "Your Name"

# Add and commit changes
!git add -A
!git commit -m "Training updates from Colab"

# Push (requires authentication)
!git push
```

**Authentication for Push:**

For private repos or push access, use Personal Access Token:

```python
# Store token securely
from getpass import getpass
token = getpass("Enter GitHub token: ")

# Push with token
!git remote set-url origin https://{token}@github.com/YOUR_USERNAME/auranet.git
!git push
```

---

## 5. VS Code ↔ Colab Workflow

### Complete Development Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT CYCLE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  VS Code (Local)              Colab (Cloud GPU)            │
│  ─────────────────           ─────────────────              │
│                                                             │
│  1. Edit code                                               │
│     └── model.py, train.py                                  │
│                                                             │
│  2. Test locally (CPU)                                      │
│     └── python train.py --debug                             │
│                                                             │
│  3. Commit & Push            4. Pull changes                │
│     └── git push      ────►     └── git pull                │
│                                                             │
│                              5. Run training (GPU)          │
│                                 └── !python train.py        │
│                                                             │
│                              6. Save checkpoints to Drive   │
│                                                             │
│  7. Pull results             ◄────  (if needed)             │
│     └── git pull                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 5.1: Local Development (VS Code)

```bash
# 1. Make changes to code
code model.py

# 2. Test locally with debug mode
python train.py --debug --epochs 1

# 3. Commit and push
git add -A
git commit -m "Updated model architecture"
git push
```

### Step 5.2: Cloud Training (Colab)

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone/Pull latest code
!git clone https://github.com/YOUR_USERNAME/auranet.git /content/auranet 2>/dev/null || (cd /content/auranet && git pull)
%cd /content/auranet

# Cell 3: Install dependencies
!pip install -q torch torchaudio librosa numpy soundfile tqdm pyyaml

# Cell 4: Set paths
import os
os.environ['CHECKPOINT_DIR'] = '/content/drive/MyDrive/AuraNet/checkpoints'
os.environ['DATASET_DIR'] = '/content/drive/MyDrive/AuraNet/datasets'

# Cell 5: Run training
!python train.py --epochs 100 --checkpoint-dir $CHECKPOINT_DIR --data-dir $DATASET_DIR
```

### Step 5.3: Retrieve Results

**Download from Drive to local:**
- Use Google Drive desktop app (sync)
- Or download manually from drive.google.com

---

## 6. File Sync Strategy

### Recommended Project Structure

```
auranet/
├── .git/                    # Git repository
├── .gitignore               # Ignore large files
├── .vscode/                 # VS Code settings
│   └── settings.json
│
├── model.py                 # ✓ Synced via Git
├── train.py                 # ✓ Synced via Git
├── dataset.py               # ✓ Synced via Git
├── loss.py                  # ✓ Synced via Git
├── infer.py                 # ✓ Synced via Git
├── config.yaml              # ✓ Synced via Git
├── requirements.txt         # ✓ Synced via Git
│
├── notebooks/               # ✓ Synced via Git
│   └── AuraNet_Training.ipynb
│
├── utils/                   # ✓ Synced via Git
│   ├── __init__.py
│   ├── stft.py
│   └── audio_utils.py
│
├── checkpoints/             # ✗ NOT in Git (too large)
│   └── best_model.pt        #   → Stored in Google Drive
│
├── datasets/                # ✗ NOT in Git (too large)
│   ├── speech/              #   → Stored in Google Drive
│   ├── music/
│   └── noise/
│
└── venv/                    # ✗ NOT in Git (local only)
```

### What Goes Where

| Content | Git (GitHub) | Google Drive | Local Only |
|---------|--------------|--------------|------------|
| Python code (.py) | ✓ | | |
| Notebooks (.ipynb) | ✓ | | |
| Config files | ✓ | | |
| requirements.txt | ✓ | | |
| Checkpoints (.pt) | | ✓ | |
| Datasets (audio) | | ✓ | |
| venv/ | | | ✓ |
| __pycache__/ | | | ✓ |

---

## 7. Running Notebooks in VS Code

### Step 7.1: Open Notebook Locally

1. Open `.ipynb` file in VS Code
2. VS Code automatically shows notebook interface
3. Select Python kernel (top right)

### Step 7.2: Run Cells Locally

- Click ▶ next to cell to run
- `Shift+Enter` to run and move to next
- `Ctrl+Enter` to run and stay

### Step 7.3: Local vs Colab Execution

**Local (VS Code):**
- ✓ Fast iteration for small tests
- ✓ Full IDE features (IntelliSense, debugging)
- ✗ No GPU (unless you have one)
- ✗ Limited by local RAM

**Colab:**
- ✓ Free GPU (T4)
- ✓ More RAM (12-25GB)
- ✗ Session limits
- ✗ No IDE features

### Step 7.4: Conditional Code for Both Environments

```python
import os

def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Use appropriate paths
if is_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = "/content/drive/MyDrive/AuraNet/datasets"
    CHECKPOINT_DIR = "/content/drive/MyDrive/AuraNet/checkpoints"
else:
    DATA_DIR = "./datasets"
    CHECKPOINT_DIR = "./checkpoints"

print(f"Running in: {'Colab' if is_colab() else 'Local'}")
print(f"Data: {DATA_DIR}")
print(f"Checkpoints: {CHECKPOINT_DIR}")
```

---

## 8. Advanced Setup (Optional)

### Option A: SSH Tunnel with ngrok (Complex)

⚠️ **Note:** This is experimental and may violate Colab ToS.

```python
# In Colab - NOT RECOMMENDED for production
!pip install colab-ssh
from colab_ssh import launch_ssh
launch_ssh(ngrok_token="YOUR_NGROK_TOKEN", password="password")
```

### Option B: Remote Jupyter Kernel

**This is NOT possible with standard Colab.** Colab runs in Google's infrastructure and doesn't expose a direct Jupyter connection.

**Alternatives:**
1. **Google Cloud AI Platform** - Run notebooks on GCP (paid)
2. **AWS SageMaker** - Similar cloud notebooks (paid)
3. **Paperspace Gradient** - Remote Jupyter with VS Code support

### Option C: Colab Enterprise

If you have Google Workspace, Colab Enterprise may offer tighter VS Code integration.

---

## 9. Limitations

### ⚠️ Important Limitations

| Limitation | Explanation | Workaround |
|------------|-------------|------------|
| **No direct VS Code → Colab** | Colab runs in browser, not as a remote kernel | Use Git to sync code |
| **Session timeouts** | Free Colab disconnects after ~90min idle | Save checkpoints frequently |
| **12-hour limit** | Sessions reset after ~12 hours | Use Drive for persistence |
| **Authentication** | Must authenticate in browser | Use GitHub tokens for automation |
| **File system resets** | `/content/` clears on disconnect | Store everything in Drive |
| **No background runs** | Closing browser stops execution | Keep tab open or use Pro+ |

### What You CAN Do

✓ Edit code in VS Code with full IDE features  
✓ Sync code via Git  
✓ Run same notebook in Colab  
✓ Store data persistently in Drive  
✓ Download checkpoints to local  

### What You CANNOT Do

✗ Connect VS Code directly to Colab runtime  
✗ Run Colab cells from VS Code  
✗ Use VS Code debugger on Colab  
✗ Keep Colab running when browser closes (free tier)  

---

## 10. Best Practices

### Code Organization

```python
# At the top of every notebook/script
import os
import sys

# Detect environment
IN_COLAB = 'google.colab' in sys.modules

# Set paths based on environment
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    
    PROJECT_ROOT = "/content/auranet"
    DATA_ROOT = "/content/drive/MyDrive/AuraNet"
    
    # Clone/pull repo
    if not os.path.exists(PROJECT_ROOT):
        os.system("git clone https://github.com/USER/auranet.git /content/auranet")
    os.chdir(PROJECT_ROOT)
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = PROJECT_ROOT

# Common paths
CHECKPOINT_DIR = os.path.join(DATA_ROOT, "checkpoints")
DATASET_DIR = os.path.join(DATA_ROOT, "datasets")
```

### Checkpoint Strategy

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save checkpoint with all training state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"💾 Checkpoint saved: {path}")

# Save every N epochs AND after each epoch to Drive
if epoch % 5 == 0 or epoch == num_epochs - 1:
    save_checkpoint(
        model, optimizer, epoch, loss,
        f"/content/drive/MyDrive/AuraNet/checkpoints/model_epoch_{epoch}.pt"
    )
```

### Resume Training

```python
def load_checkpoint(model, optimizer, path):
    """Load checkpoint and resume training."""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Resumed from epoch {start_epoch}")
        return start_epoch
    return 0

# At training start
start_epoch = load_checkpoint(
    model, optimizer,
    "/content/drive/MyDrive/AuraNet/checkpoints/latest.pt"
)
```

### Git Workflow Checklist

```bash
# Before pushing to GitHub
□ Test code locally (python train.py --debug)
□ Update requirements.txt if needed
□ Check .gitignore (no large files)
□ Write meaningful commit message

# Before running in Colab
□ Pull latest changes (!git pull)
□ Verify GPU is enabled
□ Mount Drive
□ Check checkpoint paths
```

### Quick Reference Commands

**VS Code Terminal:**
```bash
# Test locally
python train.py --debug --epochs 1

# Push to GitHub
git add -A && git commit -m "message" && git push
```

**Colab Cell:**
```python
# Full setup cell
!git clone https://github.com/USER/auranet.git /content/auranet 2>/dev/null || (cd /content/auranet && git pull)
%cd /content/auranet
from google.colab import drive; drive.mount('/content/drive')
!pip install -q -r requirements.txt
```

---

## 📚 Additional Resources

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [PyTorch Checkpointing](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No GPU" in Colab | Runtime → Change runtime type → GPU |
| Drive not mounting | Re-run mount cell, re-authenticate |
| Git push fails | Check token permissions, use HTTPS |
| Module not found | Run `!pip install` in Colab |
| Checkpoint not loading | Verify path, check Drive is mounted |
| Out of memory | Reduce batch size, use gradient accumulation |

---

*Last updated: March 2026*
