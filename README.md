# 26FW-FinML
Course materials of Financial Machine Learning part

## Configuration Management with Hydra

This project uses [Hydra](https://hydra.cc/) for flexible configuration management. All configuration files are located in the `config/` directory.

### Quick Start

```bash
# Run with default configuration (GRU model with MSE loss)
python -m src.main

# Override parameters from command line
python -m src.main seed=123 lr=0.001 batch_size=256

# Use a pre-configured experiment
python -m src.main --config-path=config/experiment --config-name=gru_experiment

# Run with different seeds (multirun)
python -m src.main --multirun seed=1,2,3,4,5
```

### Configuration Structure

- `config/config.yaml` - Main configuration (GRU + MSE by default)
- `config/model/base.yaml` - GRU model architecture
- `config/experiment/` - Example experiment configs

### Key Examples

```bash
# Change hyperparameters
python -m src.main lr=0.0005 batch_size=512 n_epochs=150

# Modify GRU architecture
python -m src.main model.gru.hidden_dim=256 model.gru.num_layers=3

# Enable Wandb logging
python -m src.main wandb=true expr_name=my_experiment
```

For detailed documentation, see [HYDRA_GUIDE.md](HYDRA_GUIDE.md).

### Extending the Configuration

The configuration is intentionally kept simple with GRU and MSE. To add more models or losses:

1. Add model configs to `config/model/base.yaml`
2. Register models in `src/models/__init__.py` model_mapper
3. Update loss handling in `src/main.py` if needed