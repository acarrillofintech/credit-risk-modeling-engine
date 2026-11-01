"""Credit scoring using logistic regression.

The model estimates a borrower's probability of default from
financial and behavioral characteristics.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int_]


@dataclass(frozen=True)
class CreditScoringMetrics:
    """Classification metrics for a probability-of-default model."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    brier_score: float


@dataclass(frozen=True)
class CreditScoringModel:
    """Trained credit-scoring model and its metadata."""

    pipeline: Pipeline
    feature_names: tuple[str, ...]
    default_threshold: float


@dataclass(frozen=True)
class CreditScoringResult:
    """Training result for a credit-scoring model."""

    model: CreditScoringModel
    metrics: CreditScoringMetrics
    training_observations: int
    testing_observations: int
    default_rate: float


def _validate_feature_matrix(
    features: object,
) -> FloatArray:
    """Validate and normalize a two-dimensional feature matrix."""

    try:
        matrix = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "features must contain numeric values."
        ) from error

    if matrix.ndim != 2:
        raise ValueError(
            "features must be a two-dimensional matrix."
        )

    if matrix.shape[0] < 4:
        raise ValueError(
            "features must contain at least four observations."
        )

    if matrix.shape[1] < 1:
        raise ValueError(
            "features must contain at least one variable."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "features must contain only finite values."
        )

    return matrix.astype(float, copy=False)


def _validate_target(
    target: object,
    expected_observations: int,
) -> IntegerArray:
    """Validate a binary default target."""

    try:
        target_array = np.asarray(target)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "target must be an array-like object."
        ) from error

    if target_array.ndim != 1:
        raise ValueError(
            "target must be one-dimensional."
        )

    if len(target_array) != expected_observations:
        raise ValueError(
            "features and target must have the same "
            "number of observations."
        )

    try:
        numeric_target = target_array.astype(float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "target must contain numeric binary values."
        ) from error

    if not np.all(np.isfinite(numeric_target)):
        raise ValueError(
            "target must contain only finite values."
        )

    unique_values = set(np.unique(numeric_target).tolist())

    if not unique_values.issubset({0.0, 1.0}):
        raise ValueError(
            "target must contain only 0 and 1."
        )

    if unique_values != {0.0, 1.0}:
        raise ValueError(
            "target must contain both default classes."
        )

    return numeric_target.astype(int)


def _validate_feature_names(
    feature_names: Sequence[str] | None,
    feature_count: int,
) -> tuple[str, ...]:
    """Validate or generate feature names."""

    if feature_names is None:
        return tuple(
            f"feature_{index + 1}"
            for index in range(feature_count)
        )

    if isinstance(feature_names, (str, bytes)):
        raise TypeError(
            "feature_names must be a sequence of strings."
        )

    normalized_names = tuple(feature_names)

    if len(normalized_names) != feature_count:
        raise ValueError(
            "feature_names must match the number of features."
        )

    if any(
        not isinstance(name, str) or not name.strip()
        for name in normalized_names
    ):
        raise ValueError(
            "Every feature name must be a non-empty string."
        )

    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError(
            "feature_names must be unique."
        )

    return normalized_names


def _validate_threshold(default_threshold: float) -> float:
    """Validate the probability threshold used for classification."""

    if (
        isinstance(default_threshold, bool)
        or not isinstance(default_threshold, Real)
    ):
        raise TypeError(
            "default_threshold must be a real number."
        )

    normalized_threshold = float(default_threshold)

    if not isfinite(normalized_threshold):
        raise ValueError(
            "default_threshold must be finite."
        )

    if not 0.0 < normalized_threshold < 1.0:
        raise ValueError(
            "default_threshold must be between 0 and 1."
        )

    return normalized_threshold


def _validate_test_size(test_size: float) -> float:
    """Validate the proportion reserved for model testing."""

    if isinstance(test_size, bool) or not isinstance(test_size, Real):
        raise TypeError(
            "test_size must be a real number."
        )

    normalized_test_size = float(test_size)

    if not isfinite(normalized_test_size):
        raise ValueError(
            "test_size must be finite."
        )

    if not 0.0 < normalized_test_size < 1.0:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    return normalized_test_size


def calculate_scoring_metrics(
    actual_defaults: object,
    predicted_defaults: object,
    default_probabilities: object,
) -> CreditScoringMetrics:
    """Calculate classification and probability-quality metrics."""

    actual = np.asarray(actual_defaults)
    predicted = np.asarray(predicted_defaults)
    probabilities = np.asarray(
        default_probabilities,
        dtype=float,
    )

    if actual.ndim != 1:
        raise ValueError(
            "actual_defaults must be one-dimensional."
        )

    if predicted.ndim != 1:
        raise ValueError(
            "predicted_defaults must be one-dimensional."
        )

    if probabilities.ndim != 1:
        raise ValueError(
            "default_probabilities must be one-dimensional."
        )

    if not (
        len(actual) == len(predicted) == len(probabilities)
    ):
        raise ValueError(
            "Metric inputs must have equal lengths."
        )

    if len(actual) == 0:
        raise ValueError(
            "Metric inputs cannot be empty."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "default_probabilities must be finite."
        )

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(
            "default_probabilities must be between 0 and 1."
        )

    if set(np.unique(actual).tolist()) != {0, 1}:
        raise ValueError(
            "actual_defaults must contain both 0 and 1."
        )

    if not set(np.unique(predicted).tolist()).issubset({0, 1}):
        raise ValueError(
            "predicted_defaults must contain only 0 and 1."
        )

    return CreditScoringMetrics(
        accuracy=float(accuracy_score(actual, predicted)),
        precision=float(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        f1_score=float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        roc_auc=float(
            roc_auc_score(actual, probabilities)
        ),
        brier_score=float(
            brier_score_loss(actual, probabilities)
        ),
    )


def train_credit_scoring_model(
    features: object,
    target: object,
    feature_names: Sequence[str] | None = None,
    test_size: float = 0.25,
    default_threshold: float = 0.50,
    random_state: int = 42,
) -> CreditScoringResult:
    """Train and evaluate a logistic credit-scoring model."""

    feature_matrix = _validate_feature_matrix(features)
    target_array = _validate_target(
        target=target,
        expected_observations=feature_matrix.shape[0],
    )
    normalized_names = _validate_feature_names(
        feature_names=feature_names,
        feature_count=feature_matrix.shape[1],
    )
    normalized_test_size = _validate_test_size(test_size)
    normalized_threshold = _validate_threshold(
        default_threshold
    )

    if isinstance(random_state, bool) or not isinstance(
        random_state,
        int,
    ):
        raise TypeError(
            "random_state must be an integer."
        )

    try:
        (
            training_features,
            testing_features,
            training_target,
            testing_target,
        ) = train_test_split(
            feature_matrix,
            target_array,
            test_size=normalized_test_size,
            random_state=random_state,
            stratify=target_array,
        )
    except ValueError as error:
        raise ValueError(
            "The dataset does not contain enough observations "
            "in each class for a stratified split."
        ) from error

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(training_features, training_target)

    default_probabilities = pipeline.predict_proba(
        testing_features
    )[:, 1]

    predicted_defaults = (
        default_probabilities >= normalized_threshold
    ).astype(int)

    metrics = calculate_scoring_metrics(
        actual_defaults=testing_target,
        predicted_defaults=predicted_defaults,
        default_probabilities=default_probabilities,
    )

    model = CreditScoringModel(
        pipeline=pipeline,
        feature_names=normalized_names,
        default_threshold=normalized_threshold,
    )

    return CreditScoringResult(
        model=model,
        metrics=metrics,
        training_observations=len(training_target),
        testing_observations=len(testing_target),
        default_rate=float(np.mean(target_array)),
    )


def predict_default_probabilities(
    model: CreditScoringModel,
    features: object,
) -> FloatArray:
    """Estimate probabilities of default for new borrowers."""

    if not isinstance(model, CreditScoringModel):
        raise TypeError(
            "model must be a CreditScoringModel."
        )

    feature_matrix = _validate_feature_matrix(features)

    if feature_matrix.shape[1] != len(model.feature_names):
        raise ValueError(
            "features must match the variables used for training."
        )

    probabilities = model.pipeline.predict_proba(
        feature_matrix
    )[:, 1]

    return np.asarray(probabilities, dtype=float)


def predict_default_classes(
    model: CreditScoringModel,
    features: object,
) -> IntegerArray:
    """Classify new borrowers using the model threshold."""

    probabilities = predict_default_probabilities(
        model=model,
        features=features,
    )

    return (
        probabilities >= model.default_threshold
    ).astype(int)


def get_model_coefficients(
    model: CreditScoringModel,
) -> dict[str, float]:
    """Return logistic-regression coefficients by feature."""

    if not isinstance(model, CreditScoringModel):
        raise TypeError(
            "model must be a CreditScoringModel."
        )

    classifier = model.pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]

    return {
        feature_name: float(coefficient)
        for feature_name, coefficient in zip(
            model.feature_names,
            coefficients,
            strict=True,
        )
    }


def main() -> None:
    """Train an illustrative credit-scoring model."""

    from sklearn.datasets import make_classification

    features, target = make_classification(
        n_samples=2_000,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        weights=[0.85, 0.15],
        class_sep=1.2,
        random_state=42,
    )

    feature_names = (
        "income_stability",
        "debt_to_income",
        "payment_history",
        "credit_utilization",
        "account_age",
    )

    result = train_credit_scoring_model(
        features=features,
        target=target,
        feature_names=feature_names,
        test_size=0.25,
        default_threshold=0.50,
        random_state=42,
    )

    print("Credit scoring with logistic regression")
    print(f"Total observations: {len(target):,}")
    print(
        f"Training observations: "
        f"{result.training_observations:,}"
    )
    print(
        f"Testing observations: "
        f"{result.testing_observations:,}"
    )
    print(f"Portfolio default rate: {result.default_rate:.2%}")

    print("\nModel performance")
    print(f"Accuracy: {result.metrics.accuracy:.4f}")
    print(f"Precision: {result.metrics.precision:.4f}")
    print(f"Recall: {result.metrics.recall:.4f}")
    print(f"F1 score: {result.metrics.f1_score:.4f}")
    print(f"ROC AUC: {result.metrics.roc_auc:.4f}")
    print(f"Brier score: {result.metrics.brier_score:.4f}")

    print("\nStandardized model coefficients")

    coefficients = get_model_coefficients(result.model)

    for feature_name, coefficient in coefficients.items():
        print(f"{feature_name}: {coefficient:+.6f}")


if __name__ == "__main__":
    main()