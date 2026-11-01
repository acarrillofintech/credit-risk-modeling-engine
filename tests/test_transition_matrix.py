"""Tests for credit-rating transition matrix models."""

import numpy as np
import pytest

from src.transition_matrix import (
    RatingMigrationResult,
    RatingTransitionModel,
    calculate_default_probability,
    calculate_transition_matrix,
    create_transition_model,
    get_transition_probability,
    project_rating_distribution,
    simulate_rating_migrations,
)


@pytest.fixture
def ratings() -> tuple[str, ...]:
    """Create representative credit-rating states."""

    return ("AAA", "AA", "A", "BBB", "BB", "B", "Default")


@pytest.fixture
def transition_matrix() -> np.ndarray:
    """Create a valid annual transition matrix."""

    return np.array(
        [
            [0.900, 0.080, 0.015, 0.004, 0.001, 0.000, 0.000],
            [0.020, 0.880, 0.075, 0.020, 0.004, 0.001, 0.000],
            [0.005, 0.040, 0.860, 0.075, 0.015, 0.004, 0.001],
            [0.000, 0.005, 0.050, 0.820, 0.090, 0.030, 0.005],
            [0.000, 0.000, 0.005, 0.060, 0.780, 0.130, 0.025],
            [0.000, 0.000, 0.000, 0.010, 0.080, 0.810, 0.100],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
        ]
    )


@pytest.fixture
def transition_model(
    ratings: tuple[str, ...],
    transition_matrix: np.ndarray,
) -> RatingTransitionModel:
    """Create a reusable transition model."""

    return create_transition_model(
        ratings=ratings,
        transition_matrix=transition_matrix,
    )


def test_create_transition_model_returns_dataclass(
    transition_model: RatingTransitionModel,
) -> None:
    """Model creation should return structured data."""

    assert isinstance(
        transition_model,
        RatingTransitionModel,
    )


def test_model_preserves_rating_order(
    transition_model: RatingTransitionModel,
    ratings: tuple[str, ...],
) -> None:
    """Rating order should match matrix rows and columns."""

    assert transition_model.ratings == ratings
    assert transition_model.default_rating == "Default"
    assert transition_model.default_index == 6


def test_transition_matrix_is_read_only(
    transition_model: RatingTransitionModel,
) -> None:
    """Validated transition probabilities should be immutable."""

    assert not transition_model.transition_matrix.flags.writeable

    with pytest.raises(ValueError):
        transition_model.transition_matrix[0, 0] = 0.50


def test_one_period_matrix_matches_original(
    transition_model: RatingTransitionModel,
) -> None:
    """A one-period projection should return the base matrix."""

    result = calculate_transition_matrix(
        model=transition_model,
        periods=1,
    )

    assert np.allclose(
        result,
        transition_model.transition_matrix,
    )


def test_zero_period_matrix_is_identity(
    transition_model: RatingTransitionModel,
) -> None:
    """Zero periods should represent no rating migration."""

    result = calculate_transition_matrix(
        model=transition_model,
        periods=0,
    )

    assert np.allclose(
        result,
        np.eye(len(transition_model.ratings)),
    )


def test_multi_period_matrix_rows_sum_to_one(
    transition_model: RatingTransitionModel,
) -> None:
    """Multi-year transition rows should remain probabilities."""

    result = calculate_transition_matrix(
        model=transition_model,
        periods=10,
    )

    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_known_one_year_transition_probability(
    transition_model: RatingTransitionModel,
) -> None:
    """BBB one-year default probability should match the matrix."""

    result = get_transition_probability(
        model=transition_model,
        initial_rating="BBB",
        final_rating="Default",
        periods=1,
    )

    assert result == pytest.approx(0.005)


def test_default_probability_uses_default_state(
    transition_model: RatingTransitionModel,
) -> None:
    """Default helper should use the configured default column."""

    result = calculate_default_probability(
        model=transition_model,
        initial_rating="B",
        periods=1,
    )

    assert result == pytest.approx(0.10)


def test_cumulative_default_probability_increases(
    transition_model: RatingTransitionModel,
) -> None:
    """Longer horizons should increase cumulative default risk."""

    one_year = calculate_default_probability(
        model=transition_model,
        initial_rating="BBB",
        periods=1,
    )
    five_year = calculate_default_probability(
        model=transition_model,
        initial_rating="BBB",
        periods=5,
    )

    assert five_year > one_year


def test_default_state_remains_in_default(
    transition_model: RatingTransitionModel,
) -> None:
    """Default is an absorbing state."""

    result = calculate_default_probability(
        model=transition_model,
        initial_rating="Default",
        periods=20,
    )

    assert result == pytest.approx(1.0)


def test_projected_distribution_sums_to_one(
    transition_model: RatingTransitionModel,
) -> None:
    """Projected portfolio weights should sum to one."""

    initial_distribution = np.array(
        [5, 10, 25, 30, 15, 15, 0],
        dtype=float,
    )

    result = project_rating_distribution(
        model=transition_model,
        initial_distribution=initial_distribution,
        periods=5,
    )

    assert result.shape == (7,)
    assert result.sum() == pytest.approx(1.0)
    assert np.all(result >= 0.0)


def test_distribution_is_normalized_automatically(
    transition_model: RatingTransitionModel,
) -> None:
    """Portfolio counts should be converted into proportions."""

    result = project_rating_distribution(
        model=transition_model,
        initial_distribution=[50, 25, 25, 0, 0, 0, 0],
        periods=0,
    )

    assert np.allclose(
        result,
        np.array([0.50, 0.25, 0.25, 0, 0, 0, 0]),
    )


def test_simulation_returns_migration_result(
    transition_model: RatingTransitionModel,
) -> None:
    """Migration simulation should return structured results."""

    initial_ratings = (
        ["AAA"] * 100
        + ["BBB"] * 500
        + ["B"] * 300
        + ["Default"] * 100
    )

    result = simulate_rating_migrations(
        model=transition_model,
        initial_ratings=initial_ratings,
        periods=3,
        random_state=42,
    )

    assert isinstance(result, RatingMigrationResult)
    assert result.observation_count == 1_000
    assert len(result.final_ratings) == 1_000
    assert sum(result.final_counts.values()) == 1_000
    assert 0.0 <= result.default_rate <= 1.0


def test_simulation_is_reproducible(
    transition_model: RatingTransitionModel,
) -> None:
    """Identical random seeds should reproduce migrations."""

    initial_ratings = ["BBB"] * 1_000

    first = simulate_rating_migrations(
        model=transition_model,
        initial_ratings=initial_ratings,
        periods=5,
        random_state=42,
    )
    second = simulate_rating_migrations(
        model=transition_model,
        initial_ratings=initial_ratings,
        periods=5,
        random_state=42,
    )

    assert first.final_ratings == second.final_ratings
    assert first.final_counts == second.final_counts


def test_zero_period_simulation_preserves_ratings(
    transition_model: RatingTransitionModel,
) -> None:
    """Zero-period simulation should make no migrations."""

    initial_ratings = ["AAA", "A", "BBB", "B", "Default"]

    result = simulate_rating_migrations(
        model=transition_model,
        initial_ratings=initial_ratings,
        periods=0,
    )

    assert result.final_ratings == tuple(initial_ratings)


def test_existing_defaults_remain_default(
    transition_model: RatingTransitionModel,
) -> None:
    """Defaulted borrowers cannot return to performing ratings."""

    result = simulate_rating_migrations(
        model=transition_model,
        initial_ratings=["Default"] * 100,
        periods=10,
        random_state=42,
    )

    assert result.default_count == 100
    assert result.default_rate == pytest.approx(1.0)


def test_ratings_cannot_be_text() -> None:
    """A single string is not a valid rating collection."""

    with pytest.raises(
        TypeError,
        match="ratings must be a sequence of strings",
    ):
        create_transition_model(
            ratings="AAA",  # type: ignore[arg-type]
            transition_matrix=np.eye(3),
        )


def test_at_least_two_rating_states_are_required() -> None:
    """A transition model needs multiple states."""

    with pytest.raises(
        ValueError,
        match="at least two states",
    ):
        create_transition_model(
            ratings=["Default"],
            transition_matrix=np.ones((1, 1)),
        )


def test_blank_rating_raises_value_error() -> None:
    """Rating labels cannot be blank."""

    with pytest.raises(
        ValueError,
        match="Every rating must be a non-empty string",
    ):
        create_transition_model(
            ratings=["A", "", "Default"],
            transition_matrix=np.eye(3),
        )


def test_duplicate_rating_raises_value_error() -> None:
    """Rating labels must be unique."""

    with pytest.raises(
        ValueError,
        match="ratings must be unique",
    ):
        create_transition_model(
            ratings=["A", "A", "Default"],
            transition_matrix=np.eye(3),
        )


def test_missing_default_rating_raises_value_error() -> None:
    """The configured default state must exist."""

    with pytest.raises(
        ValueError,
        match="default_rating must be included",
    ):
        create_transition_model(
            ratings=["A", "B"],
            transition_matrix=np.eye(2),
        )


def test_non_numeric_matrix_raises_type_error(
    ratings: tuple[str, ...],
) -> None:
    """Transition probabilities must be numeric."""

    with pytest.raises(
        TypeError,
        match="must contain numeric values",
    ):
        create_transition_model(
            ratings=ratings,
            transition_matrix=[["invalid"] * 7] * 7,
        )


def test_one_dimensional_matrix_raises_value_error(
    ratings: tuple[str, ...],
) -> None:
    """Transition matrix must have rows and columns."""

    with pytest.raises(
        ValueError,
        match="must be two-dimensional",
    ):
        create_transition_model(
            ratings=ratings,
            transition_matrix=np.ones(7),
        )


def test_wrong_matrix_shape_raises_value_error(
    ratings: tuple[str, ...],
) -> None:
    """Matrix dimensions must match rating states."""

    with pytest.raises(
        ValueError,
        match="must be square and match",
    ):
        create_transition_model(
            ratings=ratings,
            transition_matrix=np.eye(6),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_non_finite_matrix_raises_value_error(
    ratings: tuple[str, ...],
    transition_matrix: np.ndarray,
    invalid_value: float,
) -> None:
    """Transition probabilities must be finite."""

    invalid_matrix = transition_matrix.copy()
    invalid_matrix[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="must contain only finite values",
    ):
        create_transition_model(
            ratings=ratings,
            transition_matrix=invalid_matrix,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [-0.01, 1.01],
)
def test_out_of_range_probability_raises_value_error(
    ratings: tuple[str, ...],
    transition_matrix: np.ndarray,
    invalid_value: float,
) -> None:
    """Probabilities must remain in the unit interval."""

    invalid_matrix = transition_matrix.copy()
    invalid_matrix[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="must be between 0 and 1",
    ):
        create_transition_model(
            ratings=ratings,
            transition_matrix=invalid_matrix,
        )


def test_rows_not_summing_to_one_raise_value_error(
    ratings: tuple[str, ...],
    transition_matrix: np.ndarray,
) -> None:
    """Every current rating needs a complete distribution."""

    invalid_matrix = transition_matrix.copy()
    invalid_matrix[0, 0] = 0.50

    with pytest.raises(
        ValueError,
        match="row must sum to 1",
    ):
        create_transition_model(
            ratings=ratings,
            transition_matrix=invalid_matrix,
        )


def test_non_absorbing_default_raises_value_error(
    ratings: tuple[str, ...],
    transition_matrix: np.ndarray,
) -> None:
    """Default state should not migrate back to performing."""

    invalid_matrix = transition_matrix.copy()
    invalid_matrix[-1] = np.array(
        [0, 0, 0, 0, 0, 0.10, 0.90]
    )

    with pytest.raises(
        ValueError,
        match="default state must be absorbing",
    ):
        create_transition_model(
            ratings=ratings,
            transition_matrix=invalid_matrix,
        )


@pytest.mark.parametrize("invalid_periods", [-1, -10])
def test_negative_periods_raise_value_error(
    transition_model: RatingTransitionModel,
    invalid_periods: int,
) -> None:
    """Transition horizons cannot be negative."""

    with pytest.raises(
        ValueError,
        match="periods cannot be negative",
    ):
        calculate_transition_matrix(
            model=transition_model,
            periods=invalid_periods,
        )


@pytest.mark.parametrize(
    "invalid_periods",
    [1.5, "5", True],
)
def test_non_integer_periods_raise_type_error(
    transition_model: RatingTransitionModel,
    invalid_periods: object,
) -> None:
    """Transition horizons must be whole periods."""

    with pytest.raises(
        TypeError,
        match="periods must be an integer",
    ):
        calculate_transition_matrix(
            model=transition_model,
            periods=invalid_periods,  # type: ignore[arg-type]
        )


def test_unknown_initial_rating_raises_value_error(
    transition_model: RatingTransitionModel,
) -> None:
    """Unknown starting ratings should be rejected."""

    with pytest.raises(
        ValueError,
        match="Unknown initial rating",
    ):
        get_transition_probability(
            model=transition_model,
            initial_rating="XYZ",
            final_rating="A",
        )


def test_unknown_final_rating_raises_value_error(
    transition_model: RatingTransitionModel,
) -> None:
    """Unknown destination ratings should be rejected."""

    with pytest.raises(
        ValueError,
        match="Unknown final rating",
    ):
        get_transition_probability(
            model=transition_model,
            initial_rating="A",
            final_rating="XYZ",
        )


def test_wrong_distribution_length_raises_value_error(
    transition_model: RatingTransitionModel,
) -> None:
    """Portfolio distribution must match all rating states."""

    with pytest.raises(
        ValueError,
        match="must match the number of ratings",
    ):
        project_rating_distribution(
            model=transition_model,
            initial_distribution=[0.5, 0.5],
        )


def test_negative_distribution_raises_value_error(
    transition_model: RatingTransitionModel,
) -> None:
    """Portfolio weights cannot be negative."""

    with pytest.raises(
        ValueError,
        match="cannot contain negative values",
    ):
        project_rating_distribution(
            model=transition_model,
            initial_distribution=[
                0.5,
                0.3,
                0.2,
                0.0,
                0.0,
                0.1,
                -0.1,
            ],
        )


def test_zero_distribution_raises_value_error(
    transition_model: RatingTransitionModel,
) -> None:
    """A portfolio distribution needs positive mass."""

    with pytest.raises(
        ValueError,
        match="must have a positive total",
    ):
        project_rating_distribution(
            model=transition_model,
            initial_distribution=np.zeros(7),
        )


def test_empty_simulation_raises_value_error(
    transition_model: RatingTransitionModel,
) -> None:
    """At least one borrower is required for simulation."""

    with pytest.raises(
        ValueError,
        match="initial_ratings cannot be empty",
    ):
        simulate_rating_migrations(
            model=transition_model,
            initial_ratings=[],
        )


def test_unknown_simulation_rating_raises_value_error(
    transition_model: RatingTransitionModel,
) -> None:
    """Simulation ratings must exist in the model."""

    with pytest.raises(
        ValueError,
        match="Unknown ratings",
    ):
        simulate_rating_migrations(
            model=transition_model,
            initial_ratings=["A", "XYZ"],
        )


def test_invalid_random_state_raises_type_error(
    transition_model: RatingTransitionModel,
) -> None:
    """Random seed must be an integer or None."""

    with pytest.raises(
        TypeError,
        match="random_state must be an integer or None",
    ):
        simulate_rating_migrations(
            model=transition_model,
            initial_ratings=["A", "BBB"],
            random_state=42.5,  # type: ignore[arg-type]
        )