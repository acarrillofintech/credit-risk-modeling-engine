"""Credit-rating transition matrix models.

This module represents credit migrations as a finite-state
Markov chain. Borrowers can improve, deteriorate, remain in the
same rating, or enter default.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RatingTransitionModel:
    """Validated credit-rating transition model."""

    ratings: tuple[str, ...]
    transition_matrix: FloatArray
    default_rating: str
    default_index: int


@dataclass(frozen=True)
class RatingMigrationResult:
    """Result of a rating-migration simulation."""

    initial_ratings: tuple[str, ...]
    final_ratings: tuple[str, ...]
    periods: int
    observation_count: int
    default_count: int
    default_rate: float
    final_counts: dict[str, int]


def _validate_periods(periods: int) -> int:
    """Validate a non-negative number of transition periods."""

    if isinstance(periods, bool) or not isinstance(
        periods,
        Integral,
    ):
        raise TypeError("periods must be an integer.")

    normalized_periods = int(periods)

    if normalized_periods < 0:
        raise ValueError("periods cannot be negative.")

    return normalized_periods


def _validate_ratings(
    ratings: Sequence[str],
) -> tuple[str, ...]:
    """Validate rating-state names."""

    if isinstance(ratings, (str, bytes)):
        raise TypeError(
            "ratings must be a sequence of strings."
        )

    normalized_ratings = tuple(ratings)

    if len(normalized_ratings) < 2:
        raise ValueError(
            "ratings must contain at least two states."
        )

    if any(
        not isinstance(rating, str) or not rating.strip()
        for rating in normalized_ratings
    ):
        raise ValueError(
            "Every rating must be a non-empty string."
        )

    if len(set(normalized_ratings)) != len(normalized_ratings):
        raise ValueError("ratings must be unique.")

    return normalized_ratings


def create_transition_model(
    ratings: Sequence[str],
    transition_matrix: object,
    default_rating: str = "Default",
    tolerance: float = 1e-10,
) -> RatingTransitionModel:
    """Create and validate a credit-rating transition model.

    Every row represents the current rating and every column
    represents the rating in the following period.
    """

    normalized_ratings = _validate_ratings(ratings)

    if not isinstance(default_rating, str) or not default_rating.strip():
        raise ValueError(
            "default_rating must be a non-empty string."
        )

    if default_rating not in normalized_ratings:
        raise ValueError(
            "default_rating must be included in ratings."
        )

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
    ):
        raise TypeError("tolerance must be a real number.")

    normalized_tolerance = float(tolerance)

    if not isfinite(normalized_tolerance):
        raise ValueError("tolerance must be finite.")

    if normalized_tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    try:
        matrix = np.asarray(
            transition_matrix,
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "transition_matrix must contain numeric values."
        ) from error

    state_count = len(normalized_ratings)

    if matrix.ndim != 2:
        raise ValueError(
            "transition_matrix must be two-dimensional."
        )

    if matrix.shape != (state_count, state_count):
        raise ValueError(
            "transition_matrix must be square and match "
            "the number of ratings."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "transition_matrix must contain only finite values."
        )

    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError(
            "Transition probabilities must be between 0 and 1."
        )

    row_sums = matrix.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=normalized_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "Every transition-matrix row must sum to 1."
        )

    default_index = normalized_ratings.index(default_rating)
    expected_default_row = np.zeros(state_count)
    expected_default_row[default_index] = 1.0

    if not np.allclose(
        matrix[default_index],
        expected_default_row,
        atol=normalized_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "The default state must be absorbing."
        )

    validated_matrix = np.array(
        matrix,
        dtype=float,
        copy=True,
    )
    validated_matrix.setflags(write=False)

    return RatingTransitionModel(
        ratings=normalized_ratings,
        transition_matrix=validated_matrix,
        default_rating=default_rating,
        default_index=default_index,
    )


def calculate_transition_matrix(
    model: RatingTransitionModel,
    periods: int = 1,
) -> FloatArray:
    """Calculate transition probabilities after several periods."""

    if not isinstance(model, RatingTransitionModel):
        raise TypeError(
            "model must be a RatingTransitionModel."
        )

    normalized_periods = _validate_periods(periods)

    result = np.linalg.matrix_power(
        model.transition_matrix,
        normalized_periods,
    )

    return np.asarray(result, dtype=float)


def get_transition_probability(
    model: RatingTransitionModel,
    initial_rating: str,
    final_rating: str,
    periods: int = 1,
) -> float:
    """Return the probability of one specific rating migration."""

    if not isinstance(model, RatingTransitionModel):
        raise TypeError(
            "model must be a RatingTransitionModel."
        )

    if initial_rating not in model.ratings:
        raise ValueError(
            f"Unknown initial rating: {initial_rating}."
        )

    if final_rating not in model.ratings:
        raise ValueError(
            f"Unknown final rating: {final_rating}."
        )

    transition_matrix = calculate_transition_matrix(
        model=model,
        periods=periods,
    )

    initial_index = model.ratings.index(initial_rating)
    final_index = model.ratings.index(final_rating)

    return float(
        transition_matrix[initial_index, final_index]
    )


def calculate_default_probability(
    model: RatingTransitionModel,
    initial_rating: str,
    periods: int = 1,
) -> float:
    """Calculate cumulative default probability by rating."""

    return get_transition_probability(
        model=model,
        initial_rating=initial_rating,
        final_rating=model.default_rating,
        periods=periods,
    )


def project_rating_distribution(
    model: RatingTransitionModel,
    initial_distribution: object,
    periods: int = 1,
) -> FloatArray:
    """Project a portfolio rating distribution through time."""

    if not isinstance(model, RatingTransitionModel):
        raise TypeError(
            "model must be a RatingTransitionModel."
        )

    try:
        distribution = np.asarray(
            initial_distribution,
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "initial_distribution must contain numeric values."
        ) from error

    if distribution.ndim != 1:
        raise ValueError(
            "initial_distribution must be one-dimensional."
        )

    if len(distribution) != len(model.ratings):
        raise ValueError(
            "initial_distribution must match the number "
            "of ratings."
        )

    if not np.all(np.isfinite(distribution)):
        raise ValueError(
            "initial_distribution must contain finite values."
        )

    if np.any(distribution < 0.0):
        raise ValueError(
            "initial_distribution cannot contain "
            "negative values."
        )

    total = float(distribution.sum())

    if total <= 0.0:
        raise ValueError(
            "initial_distribution must have a positive total."
        )

    normalized_distribution = distribution / total
    transition_matrix = calculate_transition_matrix(
        model=model,
        periods=periods,
    )

    projected_distribution = (
        normalized_distribution @ transition_matrix
    )

    return np.asarray(projected_distribution, dtype=float)


def simulate_rating_migrations(
    model: RatingTransitionModel,
    initial_ratings: Sequence[str],
    periods: int = 1,
    random_state: int | None = 42,
) -> RatingMigrationResult:
    """Simulate borrower rating migrations through time."""

    if not isinstance(model, RatingTransitionModel):
        raise TypeError(
            "model must be a RatingTransitionModel."
        )

    if isinstance(initial_ratings, (str, bytes)):
        raise TypeError(
            "initial_ratings must be a sequence of ratings."
        )

    normalized_initial_ratings = tuple(initial_ratings)

    if not normalized_initial_ratings:
        raise ValueError(
            "initial_ratings cannot be empty."
        )

    unknown_ratings = sorted(
        {
            rating
            for rating in normalized_initial_ratings
            if rating not in model.ratings
        }
    )

    if unknown_ratings:
        raise ValueError(
            "Unknown ratings: "
            + ", ".join(str(value) for value in unknown_ratings)
            + "."
        )

    normalized_periods = _validate_periods(periods)

    if (
        random_state is not None
        and (
            isinstance(random_state, bool)
            or not isinstance(random_state, Integral)
        )
    ):
        raise TypeError(
            "random_state must be an integer or None."
        )

    generator = np.random.default_rng(random_state)

    current_indices = np.array(
        [
            model.ratings.index(rating)
            for rating in normalized_initial_ratings
        ],
        dtype=int,
    )

    state_indices = np.arange(len(model.ratings))

    for _ in range(normalized_periods):
        next_indices = np.empty_like(current_indices)

        for state_index in state_indices:
            positions = np.flatnonzero(
                current_indices == state_index
            )

            if len(positions) == 0:
                continue

            next_indices[positions] = generator.choice(
                state_indices,
                size=len(positions),
                p=model.transition_matrix[state_index],
            )

        current_indices = next_indices

    final_ratings = tuple(
        model.ratings[index]
        for index in current_indices
    )

    final_counts = {
        rating: final_ratings.count(rating)
        for rating in model.ratings
    }

    default_count = final_counts[model.default_rating]
    observation_count = len(final_ratings)

    return RatingMigrationResult(
        initial_ratings=normalized_initial_ratings,
        final_ratings=final_ratings,
        periods=normalized_periods,
        observation_count=observation_count,
        default_count=default_count,
        default_rate=default_count / observation_count,
        final_counts=final_counts,
    )


def main() -> None:
    """Run an illustrative credit-rating migration analysis."""

    ratings = (
        "AAA",
        "AA",
        "A",
        "BBB",
        "BB",
        "B",
        "CCC",
        "Default",
    )

    transition_matrix = np.array(
        [
            [0.900, 0.080, 0.015, 0.004, 0.001, 0.000, 0.000, 0.000],
            [0.020, 0.880, 0.075, 0.020, 0.004, 0.001, 0.000, 0.000],
            [0.005, 0.040, 0.860, 0.075, 0.015, 0.004, 0.001, 0.000],
            [0.000, 0.005, 0.050, 0.820, 0.090, 0.025, 0.005, 0.005],
            [0.000, 0.000, 0.005, 0.060, 0.780, 0.110, 0.025, 0.020],
            [0.000, 0.000, 0.000, 0.010, 0.080, 0.720, 0.120, 0.070],
            [0.000, 0.000, 0.000, 0.000, 0.010, 0.040, 0.600, 0.350],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
        ]
    )

    model = create_transition_model(
        ratings=ratings,
        transition_matrix=transition_matrix,
    )

    print("Credit-rating transition model")
    print(f"Rating states: {', '.join(model.ratings)}")

    print("\nOne-year default probabilities")

    for rating in model.ratings[:-1]:
        probability = calculate_default_probability(
            model=model,
            initial_rating=rating,
            periods=1,
        )
        print(f"{rating:>3}: {probability:.2%}")

    print("\nFive-year cumulative default probabilities")

    for rating in model.ratings[:-1]:
        probability = calculate_default_probability(
            model=model,
            initial_rating=rating,
            periods=5,
        )
        print(f"{rating:>3}: {probability:.2%}")

    initial_distribution = np.array(
        [0.05, 0.10, 0.25, 0.30, 0.15, 0.10, 0.05, 0.00]
    )

    projected_distribution = project_rating_distribution(
        model=model,
        initial_distribution=initial_distribution,
        periods=5,
    )

    print("\nProjected rating distribution after five years")

    for rating, probability in zip(
        model.ratings,
        projected_distribution,
        strict=True,
    ):
        print(f"{rating:>7}: {probability:.2%}")


if __name__ == "__main__":
    main()