# GRPO Directory Structure

This directory has been reorganized for better maintainability and clarity.

## Directory Structure

```
GRPO/
├── core/                    # Core functionality modules
│   ├── agents.py           # User profile agents
│   ├── data.py             # Data loading utilities
│   ├── recallers.py        # Recaller implementations
│   ├── utils.py            # General utility functions
│   └── constant.py         # Constant definitions
│
├── models/                  # Model implementations
│   ├── main.py             # Main training entry point
│   ├── main_pure.py        # Pure SFT implementation
│   ├── main_pure_v2.py     # Pure SFT v2 implementation
│   ├── main_soft.py        # Soft GRPO implementation
│   ├── main_trl.py         # TRL implementation
│   ├── soft_model.py       # Soft model implementation
│   ├── soft_utils.py       # Soft-related utilities
│   └── soft_grpo.py        # Soft GRPO algorithm
│
├── trainers/                # Training utilities
│   ├── trl_trainer.py      # TRL trainer
│   └── soft_sft_trainer.py # Soft SFT trainer
│
├── baselines/               # Baseline implementations
│   ├── baseline_cem.py     # CEM baseline
│   ├── baseline_pg.py      # Policy Gradient baseline
│   └── cem_utils.py        # CEM utility functions
│
├── scripts/                 # Execution scripts
│   ├── train/               # Training scripts
│   │   ├── train.sh
│   │   ├── train_pure.sh
│   │   ├── train_soft.sh
│   │   ├── sft.sh
│   │   ├── soft_sft.sh
│   │   └── pure_sft.sh
│   │
│   ├── test/                # Testing scripts
│   │   ├── test.sh
│   │   ├── test_sft.sh
│   │   ├── test_rl.sh
│   │   ├── test_recaller.sh
│   │   └── test_baseline_pairs.sh
│   │
│   └── run/                 # Run scripts
│       ├── run_cem_baseline.sh
│       ├── run_pg_baseline.sh
│       ├── run_soft_grpo.sh
│       └── run_soft_sft.sh
│
├── configs/                 # Configuration files
│   ├── acc.yaml
│   ├── pure_acc.yaml
│   └── soft_acc.yaml
│
├── docs/                    # Documentation
│   ├── README_pure.md
│   ├── README_SofT_GRPO.md
│   ├── CEM_README.md
│   └── ...
│
├── data/                    # Data directories
│   ├── pure_models/        # Pure model outputs
│   └── recaller_metrics_data/  # Recaller metrics data
│
└── tests/                   # Test code
    ├── test_cem_demo.py
    ├── analyze_label_noise.py
    └── generate_recaller_metrics_data.py
```

## Import Paths

After reorganization, import paths have been updated:

- **Core modules**: `from GRPO.core.agents import ...`
- **Models**: `from GRPO.models.main import ...`
- **Trainers**: `from GRPO.trainers.trl_trainer import ...`
- **Baselines**: `from GRPO.baselines.baseline_cem import ...`

## Script Usage

All scripts have been moved to `scripts/` subdirectories:
- Training scripts: `GRPO/scripts/train/`
- Testing scripts: `GRPO/scripts/test/`
- Run scripts: `GRPO/scripts/run/`

Script paths in shell scripts have been updated to reflect the new structure.

## Configuration Files

All YAML configuration files are now in `configs/`:
- `GRPO/configs/acc.yaml`
- `GRPO/configs/pure_acc.yaml`
- `GRPO/configs/soft_acc.yaml`
