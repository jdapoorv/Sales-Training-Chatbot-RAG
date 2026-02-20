# Terminal Command Log

This file tracks the important setup and execution commands.

```bash
# Initialize git repository
git init && git branch -M main

# Set up virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v

# Make initial git commit
git add -A
git commit -m "feat: initial implementation of sales call copilot CLI"

# Ingest sample transcripts (example)
python cli.py
# Then at the prompt:
#   ingest a new call transcript from samples/enterprise-prospect-01.txt
#   ingest a new call transcript from samples/smb-renewal-02.txt
#   ingest a new call transcript from samples/competitive-deal-03.txt

# --- Python 3.12 fix (ChromaDB compatibility) ---
# Install pyenv
brew install pyenv

# Add to shell profile (~/.zshrc or ~/.bash_profile):
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install Python 3.12 and pin it to this project
pyenv install 3.12.9
pyenv local 3.12.9   # creates .python-version file

# Recreate the venv with Python 3.12
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
