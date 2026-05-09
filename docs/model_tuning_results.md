# Model Tuning Results

This tuning check compared the current manually configured Random Forest models against RandomizedSearchCV-tuned Random Forest models.

## Setup

- Data source: `data/Datasets_Procurement_Cleaned_20260210_210209.xlsx`
- Coordinate source: `data/Private_coordinates.xlsx`
- Model-ready rows: 5,066
- Target balance for `is_late`:
  - On time: 45.42%
  - Late: 54.58%
- Split: 80% training, 20% testing
- Random state: 42
- Classification split: stratified by `is_late`
- Tuning method: `RandomizedSearchCV`
- Cross-validation: 3-fold CV
- Search iterations: 20

## Hyperparameters Searched

```python
{
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [None, 5, 10, 20, 30, 40],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", None],
    "model__bootstrap": [True],
}
```

## Regression Results

Target: `late_days`

| Metric | Current Random Forest | Tuned Random Forest | Change |
|---|---:|---:|---:|
| MAE | 8.1113 | 9.0785 | +0.9672 |
| RMSE | 16.9815 | 18.0221 | +1.0406 |
| R2 | 0.6427 | 0.5976 | -0.0451 |

Best CV RMSE from tuning: 16.9633

Best regression parameters:

```python
{
    "model__n_estimators": 200,
    "model__min_samples_split": 2,
    "model__min_samples_leaf": 2,
    "model__max_features": "log2",
    "model__max_depth": 20,
    "model__bootstrap": True,
}
```

## Classification Results

Target: `is_late`

| Metric | Current Random Forest | Tuned Random Forest | Change |
|---|---:|---:|---:|
| Accuracy | 0.8452 | 0.8343 | -0.0108 |
| ROC-AUC | 0.9167 | 0.9164 | -0.0003 |
| Precision | 0.8694 | 0.8625 | -0.0069 |
| Recall | 0.8427 | 0.8282 | -0.0145 |
| F1 | 0.8558 | 0.8450 | -0.0108 |

Best CV ROC-AUC from tuning: 0.9127

Best classification parameters:

```python
{
    "model__n_estimators": 500,
    "model__min_samples_split": 10,
    "model__min_samples_leaf": 1,
    "model__max_features": None,
    "model__max_depth": 20,
    "model__bootstrap": True,
}
```

## Conclusion

The tuned models did not improve held-out test performance. The current manually configured Random Forest models performed better for regression and were slightly better for classification. Based on this run, the current model settings should be retained.

For the final report, this can be described as a hyperparameter tuning experiment that was conducted to test whether model performance could be improved. Since tuning did not improve the validation results, the original Random Forest configuration was kept.
