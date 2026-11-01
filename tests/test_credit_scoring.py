"""Tests for the logistic-regression credit-scoring model."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.credit_scoring import (
    CreditScoringMetrics,
    CreditScoringModel,
    CreditScoringResult,
    calculate_scoring_metrics,
    get_model_coefficients,
    predict_default_classes,
    predict_default_probabilities,
    train_credit_scoring_model,
)


@pytest.fixture
def scoring_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible synthetic credit dataset."""

    return make_classification(
        n_samples=800,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        weights=[0.80, 0.20],
        class_sep=1.3,
        random_state=42,
    )


@pytest.fixture
def trained_result(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
) -> CreditScoringResult:
    """Train a reusable model for prediction tests."""

    features, target = scoring_dataset

    return train_credit_scoring_model(
        features=features,
        target=target,
        feature_names=(
            "income",
            "debt_ratio",
            "payment_history",
            "credit_utilization",
            "account_age",
        ),
        test_size=0.25,
        random_state=42,
    )


def test_training_returns_credit_scoring_result(
    trained_result: CreditScoringResult,
) -> None:
    """Training should return a structured result."""

    assert isinstance(trained_result, CreditScoringResult)
    assert isinstance(
        trained_result.model,
        CreditScoringModel,
    )
    assert isinstance(
        trained_result.metrics,
        CreditScoringMetrics,
    )


def test_training_and_testing_observation_counts(
    trained_result: CreditScoringResult,
) -> None:
    """Train and test samples should preserve all observations."""

    assert trained_result.training_observations == 600
    assert trained_result.testing_observations == 200
    assert (
        trained_result.training_observations
        + trained_result.testing_observations
        == 800
    )


def test_default_rate_matches_target(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
    trained_result: CreditScoringResult,
) -> None:
    """Reported default rate should match the complete dataset."""

    _, target = scoring_dataset

    assert trained_result.default_rate == pytest.approx(
        np.mean(target)
    )


def test_feature_names_are_preserved(
    trained_result: CreditScoringResult,
) -> None:
    """Model should preserve the supplied feature names."""

    assert trained_result.model.feature_names == (
        "income",
        "debt_ratio",
        "payment_history",
        "credit_utilization",
        "account_age",
    )


def test_default_feature_names_are_generated(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Feature names should be generated when omitted."""

    features, target = scoring_dataset

    result = train_credit_scoring_model(
        features=features,
        target=target,
    )

    assert result.model.feature_names == (
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
    )


def test_all_model_metrics_are_probabilities(
    trained_result: CreditScoringResult,
) -> None:
    """Classification metrics should remain between zero and one."""

    metrics = trained_result.metrics

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.brier_score <= 1.0


def test_model_has_meaningful_discrimination(
    trained_result: CreditScoringResult,
) -> None:
    """Synthetic model should discriminate better than chance."""

    assert trained_result.metrics.roc_auc > 0.75


def test_predict_probabilities_returns_valid_values(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
    trained_result: CreditScoringResult,
) -> None:
    """Predicted default probabilities should be valid."""

    features, _ = scoring_dataset

    probabilities = predict_default_probabilities(
        model=trained_result.model,
        features=features[:10],
    )

    assert probabilities.shape == (10,)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_predict_classes_returns_binary_values(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
    trained_result: CreditScoringResult,
) -> None:
    """Predicted classes should contain only zero and one."""

    features, _ = scoring_dataset

    predictions = predict_default_classes(
        model=trained_result.model,
        features=features[:10],
    )

    assert predictions.shape == (10,)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_model_coefficients_match_features(
    trained_result: CreditScoringResult,
) -> None:
    """Every feature should have one logistic coefficient."""

    coefficients = get_model_coefficients(
        trained_result.model
    )

    assert len(coefficients) == 5
    assert set(coefficients) == set(
        trained_result.model.feature_names
    )
    assert all(
        np.isfinite(value)
        for value in coefficients.values()
    )


def test_calculate_scoring_metrics_known_values() -> None:
    """Metrics should be calculated from known predictions."""

    result = calculate_scoring_metrics(
        actual_defaults=np.array([0, 0, 1, 1]),
        predicted_defaults=np.array([0, 0, 0, 1]),
        default_probabilities=np.array(
            [0.10, 0.20, 0.40, 0.90]
        ),
    )

    assert result.accuracy == pytest.approx(0.75)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(0.50)
    assert result.f1_score == pytest.approx(2.0 / 3.0)
    assert result.roc_auc == pytest.approx(1.0)


def test_training_is_reproducible(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """The same random seed should produce equal metrics."""

    features, target = scoring_dataset

    first = train_credit_scoring_model(
        features=features,
        target=target,
        random_state=42,
    )
    second = train_credit_scoring_model(
        features=features,
        target=target,
        random_state=42,
    )

    assert first.metrics == second.metrics


def test_one_dimensional_features_raise_value_error() -> None:
    """Feature data must be a matrix."""

    with pytest.raises(
        ValueError,
        match="features must be a two-dimensional matrix",
    ):
        train_credit_scoring_model(
            features=np.array([1.0, 2.0, 3.0, 4.0]),
            target=np.array([0, 0, 1, 1]),
        )


def test_too_few_observations_raise_value_error() -> None:
    """At least four observations should be supplied."""

    with pytest.raises(
        ValueError,
        match="at least four observations",
    ):
        train_credit_scoring_model(
            features=np.ones((3, 2)),
            target=np.array([0, 1, 0]),
        )


def test_empty_feature_columns_raise_value_error() -> None:
    """The model requires at least one explanatory variable."""

    with pytest.raises(
        ValueError,
        match="at least one variable",
    ):
        train_credit_scoring_model(
            features=np.empty((10, 0)),
            target=np.array([0, 1] * 5),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_non_finite_features_raise_value_error(
    invalid_value: float,
) -> None:
    """Feature values must be finite."""

    features = np.ones((10, 2))
    features[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="features must contain only finite values",
    ):
        train_credit_scoring_model(
            features=features,
            target=np.array([0, 1] * 5),
        )


def test_non_numeric_features_raise_type_error() -> None:
    """Feature values must be numeric."""

    with pytest.raises(
        TypeError,
        match="features must contain numeric values",
    ):
        train_credit_scoring_model(
            features=[
                ["good", 1.0],
                ["bad", 2.0],
                ["good", 3.0],
                ["bad", 4.0],
            ],
            target=np.array([0, 1, 0, 1]),
        )


def test_two_dimensional_target_raises_value_error() -> None:
    """Target must be a one-dimensional vector."""

    with pytest.raises(
        ValueError,
        match="target must be one-dimensional",
    ):
        train_credit_scoring_model(
            features=np.ones((10, 2)),
            target=np.array([[0], [1]] * 5),
        )


def test_target_length_mismatch_raises_value_error() -> None:
    """Features and targets require matching row counts."""

    with pytest.raises(
        ValueError,
        match="must have the same number of observations",
    ):
        train_credit_scoring_model(
            features=np.ones((10, 2)),
            target=np.array([0, 1, 0, 1]),
        )


@pytest.mark.parametrize(
    "invalid_target",
    [
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([-1, 0, 1, -1, 0, 1]),
    ],
)
def test_non_binary_target_raises_value_error(
    invalid_target: np.ndarray,
) -> None:
    """Target should contain only default indicators."""

    with pytest.raises(
        ValueError,
        match="target must contain only 0 and 1",
    ):
        train_credit_scoring_model(
            features=np.ones((6, 2)),
            target=invalid_target,
        )


def test_single_target_class_raises_value_error() -> None:
    """Training requires both performing and defaulted borrowers."""

    with pytest.raises(
        ValueError,
        match="target must contain both default classes",
    ):
        train_credit_scoring_model(
            features=np.ones((10, 2)),
            target=np.zeros(10),
        )


@pytest.mark.parametrize(
    "test_size",
    [-0.10, 0.0, 1.0, 1.10],
)
def test_invalid_test_size_raises_value_error(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
    test_size: float,
) -> None:
    """Test size must be strictly between zero and one."""

    features, target = scoring_dataset

    with pytest.raises(
        ValueError,
        match="test_size must be between 0 and 1",
    ):
        train_credit_scoring_model(
            features=features,
            target=target,
            test_size=test_size,
        )


@pytest.mark.parametrize(
    "default_threshold",
    [-0.10, 0.0, 1.0, 1.10],
)
def test_invalid_threshold_raises_value_error(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
    default_threshold: float,
) -> None:
    """Classification threshold must be inside the unit interval."""

    features, target = scoring_dataset

    with pytest.raises(
        ValueError,
        match="default_threshold must be between 0 and 1",
    ):
        train_credit_scoring_model(
            features=features,
            target=target,
            default_threshold=default_threshold,
        )


def test_duplicate_feature_names_raise_value_error(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Feature names must be unique."""

    features, target = scoring_dataset

    with pytest.raises(
        ValueError,
        match="feature_names must be unique",
    ):
        train_credit_scoring_model(
            features=features,
            target=target,
            feature_names=("a", "b", "c", "d", "a"),
        )


def test_feature_name_count_mismatch_raises_value_error(
    scoring_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Feature-name count must match matrix columns."""

    features, target = scoring_dataset

    with pytest.raises(
        ValueError,
        match="must match the number of features",
    ):
        train_credit_scoring_model(
            features=features,
            target=target,
            feature_names=("a", "b"),
        )


def test_wrong_prediction_feature_count_raises_value_error(
    trained_result: CreditScoringResult,
) -> None:
    """Prediction data must match the trained model."""

    with pytest.raises(
        ValueError,
        match="must match the variables used for training",
    ):
        predict_default_probabilities(
            model=trained_result.model,
            features=np.ones((10, 3)),
        )


def test_invalid_prediction_model_raises_type_error() -> None:
    """Prediction requires a CreditScoringModel."""

    with pytest.raises(
        TypeError,
        match="model must be a CreditScoringModel",
    ):
        predict_default_probabilities(
            model="invalid",  # type: ignore[arg-type]
            features=np.ones((10, 2)),
        )


def test_metric_length_mismatch_raises_value_error() -> None:
    """Metric inputs must have equal lengths."""

    with pytest.raises(
        ValueError,
        match="Metric inputs must have equal lengths",
    ):
        calculate_scoring_metrics(
            actual_defaults=np.array([0, 1]),
            predicted_defaults=np.array([0]),
            default_probabilities=np.array([0.1, 0.9]),
        )


def test_invalid_metric_probabilities_raise_value_error() -> None:
    """Default probabilities cannot exceed valid boundaries."""

    with pytest.raises(
        ValueError,
        match="must be between 0 and 1",
    ):
        calculate_scoring_metrics(
            actual_defaults=np.array([0, 1]),
            predicted_defaults=np.array([0, 1]),
            default_probabilities=np.array([-0.1, 1.1]),
        )