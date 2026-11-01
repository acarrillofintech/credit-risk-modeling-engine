"""Monte Carlo simulation of correlated credit defaults.

The module uses a one-factor Gaussian credit model. Borrowers are
affected by both:

- a common systematic economic factor;
- an individual idiosyncratic factor.

Asset correlation controls how strongly defaults move together.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int_]


@dataclass(frozen=True)
class CreditPortfolio:
    """Validated credit portfolio used by the simulation."""

    identifiers: tuple[str, ...]
    probabilities_of_default: FloatArray
    losses_given_default: FloatArray
    exposures_at_default: FloatArray
    exposure_count: int
    total_exposure: float


@dataclass(frozen=True)
class DefaultSimulationResult:
    """Results from a correlated-default Monte Carlo simulation."""

    number_of_simulations: int
    confidence_level: float
    asset_correlation: float
    analytical_expected_loss: float
    simulated_expected_loss: float
    loss_standard_deviation: float
    value_at_risk: float
    expected_shortfall: float
    economic_capital: float
    maximum_simulated_loss: float
    average_default_count: float
    maximum_default_count: int
    loss_distribution: FloatArray
    default_count_distribution: IntegerArray


def _to_numeric_vector(
    values: object,
    parameter_name: str,
) -> FloatArray:
    """Convert input data into a finite one-dimensional array."""

    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{parameter_name} must contain numeric values."
        ) from error

    if vector.ndim != 1:
        raise ValueError(
            f"{parameter_name} must be one-dimensional."
        )

    if len(vector) == 0:
        raise ValueError(
            f"{parameter_name} cannot be empty."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{parameter_name} must contain only finite values."
        )

    return np.asarray(vector, dtype=float)


def _validate_identifiers(
    identifiers: Sequence[str] | None,
    exposure_count: int,
) -> tuple[str, ...]:
    """Validate or generate exposure identifiers."""

    if identifiers is None:
        return tuple(
            f"Exposure_{index + 1}"
            for index in range(exposure_count)
        )

    if isinstance(identifiers, (str, bytes)):
        raise TypeError(
            "identifiers must be a sequence of strings."
        )

    normalized_identifiers = tuple(identifiers)

    if len(normalized_identifiers) != exposure_count:
        raise ValueError(
            "identifiers must match the number of exposures."
        )

    if any(
        not isinstance(identifier, str)
        or not identifier.strip()
        for identifier in normalized_identifiers
    ):
        raise ValueError(
            "Every identifier must be a non-empty string."
        )

    if len(set(normalized_identifiers)) != exposure_count:
        raise ValueError(
            "identifiers must be unique."
        )

    return normalized_identifiers


def _read_only_copy(values: FloatArray) -> FloatArray:
    """Create a protected copy of a numeric array."""

    copied_values = np.array(
        values,
        dtype=float,
        copy=True,
    )
    copied_values.setflags(write=False)

    return copied_values


def create_credit_portfolio(
    probabilities_of_default: object,
    losses_given_default: object,
    exposures_at_default: object,
    identifiers: Sequence[str] | None = None,
) -> CreditPortfolio:
    """Create and validate a portfolio for default simulation."""

    probabilities = _to_numeric_vector(
        probabilities_of_default,
        "probabilities_of_default",
    )
    losses = _to_numeric_vector(
        losses_given_default,
        "losses_given_default",
    )
    exposures = _to_numeric_vector(
        exposures_at_default,
        "exposures_at_default",
    )

    exposure_count = len(probabilities)

    if not (
        len(losses) == exposure_count
        and len(exposures) == exposure_count
    ):
        raise ValueError(
            "PD, LGD, and EAD vectors must have equal lengths."
        )

    if np.any(
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ):
        raise ValueError(
            "Probabilities of default must be between 0 and 1."
        )

    if np.any((losses < 0.0) | (losses > 1.0)):
        raise ValueError(
            "Losses given default must be between 0 and 1."
        )

    if np.any(exposures < 0.0):
        raise ValueError(
            "Exposures at default cannot be negative."
        )

    normalized_identifiers = _validate_identifiers(
        identifiers=identifiers,
        exposure_count=exposure_count,
    )

    return CreditPortfolio(
        identifiers=normalized_identifiers,
        probabilities_of_default=_read_only_copy(
            probabilities
        ),
        losses_given_default=_read_only_copy(losses),
        exposures_at_default=_read_only_copy(exposures),
        exposure_count=exposure_count,
        total_exposure=float(exposures.sum()),
    )


def calculate_analytical_expected_loss(
    portfolio: CreditPortfolio,
) -> float:
    """Calculate portfolio expected loss as PD × LGD × EAD."""

    if not isinstance(portfolio, CreditPortfolio):
        raise TypeError(
            "portfolio must be a CreditPortfolio."
        )

    expected_losses = (
        portfolio.probabilities_of_default
        * portfolio.losses_given_default
        * portfolio.exposures_at_default
    )

    return float(expected_losses.sum())


def _validate_number_of_simulations(
    number_of_simulations: int,
) -> int:
    """Validate the Monte Carlo simulation count."""

    if isinstance(number_of_simulations, bool) or not isinstance(
        number_of_simulations,
        Integral,
    ):
        raise TypeError(
            "number_of_simulations must be an integer."
        )

    normalized_count = int(number_of_simulations)

    if normalized_count < 2:
        raise ValueError(
            "number_of_simulations must be at least 2."
        )

    return normalized_count


def _validate_unit_interval(
    value: float,
    parameter_name: str,
    include_zero: bool,
) -> float:
    """Validate parameters constrained to the unit interval."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{parameter_name} must be a real number."
        )

    normalized_value = float(value)

    if not isfinite(normalized_value):
        raise ValueError(
            f"{parameter_name} must be finite."
        )

    valid_lower_bound = (
        normalized_value >= 0.0
        if include_zero
        else normalized_value > 0.0
    )

    if not valid_lower_bound or normalized_value >= 1.0:
        lower_symbol = "[" if include_zero else "("

        raise ValueError(
            f"{parameter_name} must be in "
            f"{lower_symbol}0, 1)."
        )

    return normalized_value


def simulate_correlated_defaults(
    portfolio: CreditPortfolio,
    number_of_simulations: int = 50_000,
    asset_correlation: float = 0.20,
    confidence_level: float = 0.99,
    random_state: int | None = 42,
) -> DefaultSimulationResult:
    """Simulate correlated defaults using a Gaussian factor model.

    The latent variable for borrower i in simulation s is:

        Z(s,i) = sqrt(rho) × Y(s)
               + sqrt(1-rho) × epsilon(s,i)

    A default occurs when the latent variable is below the normal
    threshold associated with the borrower's PD.
    """

    if not isinstance(portfolio, CreditPortfolio):
        raise TypeError(
            "portfolio must be a CreditPortfolio."
        )

    simulation_count = _validate_number_of_simulations(
        number_of_simulations
    )
    correlation = _validate_unit_interval(
        value=asset_correlation,
        parameter_name="asset_correlation",
        include_zero=True,
    )
    confidence = _validate_unit_interval(
        value=confidence_level,
        parameter_name="confidence_level",
        include_zero=False,
    )

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

    systematic_factors = generator.standard_normal(
        size=(simulation_count, 1)
    )
    idiosyncratic_factors = generator.standard_normal(
        size=(
            simulation_count,
            portfolio.exposure_count,
        )
    )

    latent_variables = (
        sqrt(correlation) * systematic_factors
        + sqrt(1.0 - correlation)
        * idiosyncratic_factors
    )

    default_thresholds = norm.ppf(
        portfolio.probabilities_of_default
    )

    default_indicators = (
        latent_variables < default_thresholds
    )

    losses_if_default = (
        portfolio.losses_given_default
        * portfolio.exposures_at_default
    )

    loss_distribution = (
        default_indicators @ losses_if_default
    ).astype(float)

    default_count_distribution = default_indicators.sum(
        axis=1
    ).astype(int)

    analytical_expected_loss = (
        calculate_analytical_expected_loss(portfolio)
    )
    simulated_expected_loss = float(
        np.mean(loss_distribution)
    )
    loss_standard_deviation = float(
        np.std(loss_distribution, ddof=1)
    )

    value_at_risk = float(
        np.quantile(
            loss_distribution,
            confidence,
            method="higher",
        )
    )

    tail_losses = loss_distribution[
        loss_distribution >= value_at_risk
    ]
    expected_shortfall = float(np.mean(tail_losses))

    economic_capital = max(
        value_at_risk - analytical_expected_loss,
        0.0,
    )

    stored_losses = np.array(
        loss_distribution,
        dtype=float,
        copy=True,
    )
    stored_losses.setflags(write=False)

    stored_default_counts = np.array(
        default_count_distribution,
        dtype=int,
        copy=True,
    )
    stored_default_counts.setflags(write=False)

    return DefaultSimulationResult(
        number_of_simulations=simulation_count,
        confidence_level=confidence,
        asset_correlation=correlation,
        analytical_expected_loss=analytical_expected_loss,
        simulated_expected_loss=simulated_expected_loss,
        loss_standard_deviation=loss_standard_deviation,
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
        economic_capital=economic_capital,
        maximum_simulated_loss=float(
            np.max(loss_distribution)
        ),
        average_default_count=float(
            np.mean(default_count_distribution)
        ),
        maximum_default_count=int(
            np.max(default_count_distribution)
        ),
        loss_distribution=stored_losses,
        default_count_distribution=stored_default_counts,
    )


def main() -> None:
    """Run an illustrative correlated-default simulation."""

    exposure_count = 100

    portfolio = create_credit_portfolio(
        probabilities_of_default=np.linspace(
            0.005,
            0.080,
            exposure_count,
        ),
        losses_given_default=np.full(
            exposure_count,
            0.45,
        ),
        exposures_at_default=np.linspace(
            100_000.0,
            1_000_000.0,
            exposure_count,
        ),
        identifiers=[
            f"Borrower_{index + 1}"
            for index in range(exposure_count)
        ],
    )

    result = simulate_correlated_defaults(
        portfolio=portfolio,
        number_of_simulations=50_000,
        asset_correlation=0.20,
        confidence_level=0.99,
        random_state=42,
    )

    print("Correlated credit-default simulation")
    print(f"Credit exposures: {portfolio.exposure_count:,}")
    print(
        f"Total exposure: ${portfolio.total_exposure:,.2f}"
    )
    print(
        f"Monte Carlo simulations: "
        f"{result.number_of_simulations:,}"
    )
    print(
        f"Asset correlation: "
        f"{result.asset_correlation:.2%}"
    )
    print(
        f"Confidence level: "
        f"{result.confidence_level:.2%}"
    )

    print("\nPortfolio credit risk")
    print(
        "Analytical expected loss: "
        f"${result.analytical_expected_loss:,.2f}"
    )
    print(
        "Simulated expected loss: "
        f"${result.simulated_expected_loss:,.2f}"
    )
    print(
        "Loss standard deviation: "
        f"${result.loss_standard_deviation:,.2f}"
    )
    print(
        f"Credit VaR: ${result.value_at_risk:,.2f}"
    )
    print(
        "Expected Shortfall: "
        f"${result.expected_shortfall:,.2f}"
    )
    print(
        f"Economic capital: ${result.economic_capital:,.2f}"
    )
    print(
        "Maximum simulated loss: "
        f"${result.maximum_simulated_loss:,.2f}"
    )

    print("\nDefault statistics")
    print(
        "Average defaults per simulation: "
        f"{result.average_default_count:.4f}"
    )
    print(
        "Maximum defaults in one simulation: "
        f"{result.maximum_default_count}"
    )


if __name__ == "__main__":
    main()