#!/usr/bin/env python3
"""Fix notebook training settings."""

import json

nb_path = '/Users/vinoth-14902/auranet/notebooks/AuraNet_Kaggle_Training.ipynb'
with open(nb_path) as f:
    nb = json.load(f)

changes = 0

for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        # Join all source lines into one string
        src = ''.join(cell.get('source', []))
        original_src = src

        # Fix learning rate
        if '"learning_rate": 0.001' in src:
            src = src.replace('"learning_rate": 0.001', '"learning_rate": 0.0001  # FIXED: was 0.001')
            changes += 1
            print(f"Cell {i}: Fixed learning_rate")

        # Fix clamp to tanh for enhanced_audio
        if 'enhanced_audio = torch.clamp(enhanced_audio, -1.0, 1.0)' in src:
            src = src.replace(
                'enhanced_audio = torch.clamp(enhanced_audio, -1.0, 1.0)',
                'enhanced_audio = torch.tanh(enhanced_audio)  # FIXED: tanh instead of clamp'
            )
            changes += 1
            print(f"Cell {i}: Fixed clamp -> tanh")

        # Fix the energy ratio clamp line
        if 'enhanced_audio = torch.clamp(enhanced_audio * energy_ratio, -1.0, 1.0)' in src:
            src = src.replace(
                'enhanced_audio = torch.clamp(enhanced_audio * energy_ratio, -1.0, 1.0)',
                'enhanced_audio = torch.tanh(enhanced_audio * energy_ratio)  # FIXED: tanh'
            )
            changes += 1
            print(f"Cell {i}: Fixed energy_ratio clamp -> tanh")

        if src != original_src:
            # Split back into lines (preserving the notebook format)
            lines = src.split('\n')
            # Restore newlines for all but last
            cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\nTotal changes: {changes}")
