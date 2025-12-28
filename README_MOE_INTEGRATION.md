# ALIGNN MoE + ZeroMAT Integration

Complete integration of ALIGNN Mixture-of-Experts with ZeroMAT framework for crystal property prediction.

## 🎯 Overview

This project integrates **ALIGNN MoE** (crystal materials) with the existing **ZeroMAT** (3D molecules) framework using a unified BLIP2-QFormer architecture.

### Key Features

- **Property-Specific Expert Routing**: Top-k ALIGNN experts selected per property
- **Frozen MoE Encoder**: No gradient updates during Stage 2 training
- **Property-Conditioned Q-Former**: Text prompts guide Q-Former attention
- **4-Loss Training**: ITC + ITM + LM + Property Prediction
- **Minimal Code Changes**: Existing ZeroMAT code fully preserved

---

## 📂 Project Structure

```
CrystalMOELLM/
├── MoE/                              # New MoE modules
│   ├── __init__.py
│   ├── alignn_extractor.py           # ALIGNN backbone (no head)
│   ├── moe_encoder.py                # Frozen MoE with routing
│   └── moe_downstream.py             # MoE downstream training
│
├── model/
│   ├── blip2qformer.py               # Modified (MoE support added)
│   ├── blip2_moe_stage2.py           # New Stage 2 trainer
│   └── ... (existing files unchanged)
│
├── data_provider/
│   ├── crystal_dataset.py            # New crystal dataset
│   ├── crystal_dm.py                 # New Lightning DataModule
│   └── ... (existing files unchanged)
│
├── scripts/
│   └── train_moe_stage2.py           # Main training script
│
├── configs/
│   └── property_list.yaml            # 12 properties config
│
├── setup.py                          # Installation script
└── README_MOE_INTEGRATION.md         # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install -e .
```

### 2. Data Preparation

Prepare your crystal data in JSON format:

```json
[
  {
    "id": "mp-1234",
    "graph_path": "data/graphs/mp-1234.bin",
    "properties": {
      "band_gap": 3.2,
      "formation_energy": -1.5,
      ...
    },
    "text": "GaN is a wurtzite structure..."
  }
]
```

Save as `data/splits/{train,val,test}.json`.

### 3. Training (Stage 2)

Assuming you have:
- Trained ALIGNN experts → `checkpoints/extractors/`
- MoE top-k config → `config/moe_topk.json`

Run:

```bash
python scripts/train_moe_stage2.py \
    --filename moe_stage2_run1 \
    --use_moe \
    --moe_extractor_dir checkpoints/extractors \
    --moe_topk_config config/moe_topk.json \
    --batch_size 32 \
    --max_epochs 50 \
    --devices 0,1
```

---

## 🔧 Key Modifications

### 1. `model/blip2qformer.py`

**Added MoE Support:**

```python
if hasattr(args, 'use_moe') and args.use_moe:
    # MoE Encoder
    self.graph_encoder = FrozenMoEEncoder(...)

    # Property heads
    self.property_heads = nn.ModuleDict({...})

    # Property prompts
    self.property_prompts = {...}
```

**Modified Forward:**

```python
# 1. Graph encoding with property routing
batch_node, batch_mask = self.graph_encoder(graph_batch, property_info['property_name'])

# 2. Property-conditioned Q-Former
prompt_text = self.property_prompts[property_info['property_name']]
query_output = self.Qformer.bert(input_ids=prompt_tokens, ...)

# 3. Property prediction loss
loss_property = F.l1_loss(prop_pred, prop_target)
```

### 2. New Components

| Component | Purpose |
|-----------|---------|
| `MoE/moe_encoder.py` | Frozen MoE with property routing |
| `MoE/alignn_extractor.py` | ALIGNN feature extractor |
| `data_provider/crystal_dm.py` | Crystal dataset & DataModule |
| `model/blip2_moe_stage2.py` | PyTorch Lightning trainer |

---

## 📊 4 Losses

| Loss | Formula | Weight |
|------|---------|--------|
| **ITC** | Graph ↔ Text contrastive | 1.0 |
| **ITM** | Graph-Text matching (optional) | 1.0 |
| **LM** | Robocrys generation | 1.0 |
| **Property** | L1(prediction, target) | 1.0 |

**Total Loss:**
```python
loss = loss_itc + loss_itm + loss_lm + loss_property
```

---

## 🎨 Property-Conditioned Q-Former

### How It Works

```
Input: Crystal graph for "band_gap" prediction

Step 1: MoE Routing
  → Top-k experts: [Expert_0, Expert_7, Expert_2]
  → Weighted combination

Step 2: Property Prompt
  → "Calculate the band gap energy:"
  → Tokenized and fed to Q-Former

Step 3: Q-Former Attention
  → Query tokens attend to:
    - Graph features (from MoE)
    - Property prompt tokens
  → Output: band gap-specific features

Step 4: Prediction
  → Property head: features → scalar
```

### Benefits

- ✅ Q-Former knows **what** to predict
- ✅ Focused attention on relevant features
- ✅ Better property-specific representations

---

## 🔬 12 Properties

```yaml
- band_gap              # Band gap energy (eV)
- formation_energy      # Formation energy (eV/atom)
- bulk_modulus          # Bulk modulus (GPa)
- shear_modulus         # Shear modulus (GPa)
- elastic_anisotropy    # Elastic anisotropy
- poisson_ratio         # Poisson ratio
- total_magnetization   # Total magnetization (μB)
- n_Egap                # n-type band gap (eV)
- p_Egap                # p-type band gap (eV)
- n_mass                # n-type effective mass
- p_mass                # p-type effective mass
- eij_max               # Max piezoelectric constant
```

---

## 📈 Expected Performance

### Stage 2 Training

```
Epoch 10/50
Train - Loss: 2.456, ITC: 0.832, LM: 1.425, Property: 0.199
Val   - Loss: 2.512, ITC: 0.845, LM: 1.458, Property: 0.209

Property Prediction MAE:
  band_gap: 0.142 eV
  formation_energy: 0.030 eV/atom
```

---

## 🛠️ Configuration

### `configs/property_list.yaml`

```yaml
properties:
  - band_gap
  - formation_energy
  ...

descriptions:
  band_gap: "Band gap energy (eV)"
  ...
```

### MoE Top-k Config (`config/moe_topk.json`)

```json
{
  "band_gap": {
    "indices": [0, 7, 2],
    "probs": [0.48, 0.35, 0.17]
  },
  ...
}
```

---

## 🔧 Advanced Usage

### Custom Property Prompts

Modify `blip2qformer.py`:

```python
self.property_prompts = {
    'band_gap': 'Predict electronic band gap in eV:',
    'formation_energy': 'Calculate thermodynamic stability:',
    ...
}
```

### Adjust Loss Weights

```python
# In training script
loss = (
    1.0 * loss_itc +
    0.5 * loss_itm +   # Reduce ITM weight
    1.0 * loss_lm +
    2.0 * loss_property  # Increase property weight
)
```

---

## ❓ FAQ

**Q: Can I use existing ZeroMAT checkpoints?**
A: Yes! MoE only activates when `--use_moe` is set.

**Q: How many GPUs do I need?**
A: 2+ GPUs recommended (batch_size=32, bf16 precision).

**Q: What if I only have 8 properties, not 12?**
A: Modify `configs/property_list.yaml` and train correspondingly.

---

## 📝 Citation

```bibtex
@article{alignn2021,
  title={Atomistic Line Graph Neural Network},
  author={Choudhary, Kamal and DeCost, Brian},
  journal={npj Computational Materials},
  year={2021}
}
```

---

## 📧 Contact

For issues: Open a GitHub issue
For questions: Check documentation first

---

**Happy Training! 🚀**
