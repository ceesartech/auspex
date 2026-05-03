# ML Models Service

Machine learning models for sports match prediction. Implements 5 model types with training pipeline, hyperparameter optimization, explainability, and inference optimization.

## Model Types

| Model | Type | Task | Key Feature |
|-------|------|------|-------------|
| XGBoost | Gradient Boosting | Match Outcome | SHAP explainability |
| LightGBM | Gradient Boosting | Match Outcome / Over-Under | Fast training |
| Neural Network | Deep Learning | Match Outcome | BatchNorm + Dropout |
| Poisson | Statistical | Score Prediction | Score matrices, BTTS, O/U |
| Dixon-Coles | Statistical | Score Prediction | Low-score correction (rho) |
| Ensemble | Meta-learner | Match Outcome | Optimized weight blending |

## Architecture

```
src/
├── models/           # Model implementations
│   ├── base_model.py         # Abstract base class
│   ├── model_config.py       # Configuration system
│   ├── xgboost_model.py      # XGBoost with SHAP
│   ├── lightgbm_model.py     # LightGBM (binary + multiclass)
│   ├── neural_network.py     # PyTorch NN
│   ├── poisson_models.py     # Poisson + Dixon-Coles
│   ├── ensemble.py           # Weighted ensemble
│   └── model_registry.py     # Version management
├── training/         # Training pipeline
│   ├── cross_validation.py   # Time-series CV
│   ├── hyperparameter_optimization.py  # Optuna
│   ├── calibration.py        # Probability calibration
│   └── train_all_models.py   # Training orchestrator
├── evaluation/       # Metrics and comparison
│   ├── metrics.py            # Accuracy, log loss, RPS, ROI
│   ├── model_comparison.py   # A/B testing + bootstrap
│   └── performance_tracker.py # Drift detection
├── inference/        # Serving
│   ├── real_time_predictor.py # Redis cache + ONNX
│   ├── batch_predictor.py     # Batch + value bets
│   └── onnx_converter.py     # ONNX conversion
├── explainability/   # Model interpretation
│   └── shap_explainer.py     # SHAP (Tree/Deep/Kernel)
└── utils/            # Utilities
    ├── data_loader.py        # Data loading + splitting
    └── feature_selector.py   # Feature selection methods
```

## Key Features

- **Time-Series CV**: Prevents data leakage with chronological splits
- **Optuna HPO**: TPE sampler with median pruner for hyperparameter search
- **Probability Calibration**: Isotonic regression / Platt scaling
- **ONNX Inference**: Sub-100ms predictions via ONNX runtime
- **Redis Caching**: Prediction caching with configurable TTL
- **Value Bet Detection**: Identifies edges where model prob > implied prob
- **SHAP Explainability**: Feature-level explanations for individual predictions
- **Drift Detection**: Monitors model performance degradation over time
- **A/B Testing**: Compare model versions on live predictions

## Running Tests

```bash
cd services/ml-models
pip install -r requirements.txt
pytest tests/ -v
```
