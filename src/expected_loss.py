"""Expected and unexpected credit-loss calculations.

This module implements fundamental credit-risk measures based on:

- Probability of Default (PD)
- Loss Given Default (LGD)
- Exposure at Default (EAD)
- Expected Loss (EL)
- Unexpected Loss (UL)
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real
from typing import Sequence


@dataclass(frozen=True)
class CreditExposure:
    """Credit-risk information for a single borrower or facility."""

    identifier: str
    probability_of_default: float
    loss_given_default: float
    exposure_at_default: float


@dataclass(frozen=True)
class CreditRiskMetrics:
    """Credit-risk metrics for a single exposure."""

    expected_loss: float
    unexpected_loss: float
    maximum_loss: float


@dataclass(frozen=True)
class PortfolioCreditRisk:
    """Aggregated credit-risk metrics for a portfolio."""

    exposure_count: int
    total_exposure: float
    expected_loss: float
    unexpected_loss: float
    maximum_loss: float


def _validate_probability(
    value: float,
    parameter_name: str,
) -> float:
    """Validate a probability expressed as a decimal."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{parameter_name} must be a real number."
        )

    normalized_value = float(value)

    if not isfinite(normalized_value):
        raise ValueError(
            f"{parameter_name} must be finite."
        )

    if not 0.0 <= normalized_value <= 1.0:
        raise ValueError(
            f"{parameter_name} must be between 0 and 1."
        )

    return normalized_value


def _validate_exposure(value: float) -> float:
    """Validate an exposure-at-default amount."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "exposure_at_default must be a real number."
        )

    normalized_value = float(value)

    if not isfinite(normalized_value):
        raise ValueError(
            "exposure_at_default must be finite."
        )

    if normalized_value < 0.0:
        raise ValueError(
            "exposure_at_default cannot be negative."
        )

    return normalized_value


def calculate_expected_loss(
    probability_of_default: float,
    loss_given_default: float,
    exposure_at_default: float,
) -> float:
    """Calculate expected credit loss.

    Expected Loss = PD × LGD × EAD
    """

    pd = _validate_probability(
        probability_of_default,
        "probability_of_default",
    )
    lgd = _validate_probability(
        loss_given_default,
        "loss_given_default",
    )
    ead = _validate_exposure(exposure_at_default)

    return pd * lgd * ead


def calculate_unexpected_loss(
    probability_of_default: float,
    loss_given_default: float,
    exposure_at_default: float,
) -> float:
    """Calculate unexpected loss using a Bernoulli default model.

    Unexpected Loss =
        LGD × EAD × sqrt(PD × (1 - PD))
    """

    pd = _validate_probability(
        probability_of_default,
        "probability_of_default",
    )
    lgd = _validate_probability(
        loss_given_default,
        "loss_given_default",
    )
    ead = _validate_exposure(exposure_at_default)

    return lgd * ead * sqrt(pd * (1.0 - pd))


def calculate_maximum_loss(
    loss_given_default: float,
    exposure_at_default: float,
) -> float:
    """Calculate the loss if default occurs."""

    lgd = _validate_probability(
        loss_given_default,
        "loss_given_default",
    )
    ead = _validate_exposure(exposure_at_default)

    return lgd * ead


def calculate_credit_risk_metrics(
    probability_of_default: float,
    loss_given_default: float,
    exposure_at_default: float,
) -> CreditRiskMetrics:
    """Calculate all fundamental metrics for one exposure."""

    expected_loss = calculate_expected_loss(
        probability_of_default=probability_of_default,
        loss_given_default=loss_given_default,
        exposure_at_default=exposure_at_default,
    )
    unexpected_loss = calculate_unexpected_loss(
        probability_of_default=probability_of_default,
        loss_given_default=loss_given_default,
        exposure_at_default=exposure_at_default,
    )
    maximum_loss = calculate_maximum_loss(
        loss_given_default=loss_given_default,
        exposure_at_default=exposure_at_default,
    )

    return CreditRiskMetrics(
        expected_loss=expected_loss,
        unexpected_loss=unexpected_loss,
        maximum_loss=maximum_loss,
    )


def calculate_portfolio_credit_risk(
    exposures: Sequence[CreditExposure],
) -> PortfolioCreditRisk:
    """Aggregate independent credit exposures.

    Portfolio expected loss is the sum of individual expected
    losses.

    Portfolio unexpected loss assumes independent defaults and
    therefore combines individual unexpected losses using the
    square root of the sum of squared values.
    """

    if isinstance(exposures, (str, bytes)):
        raise TypeError(
            "exposures must be a sequence of CreditExposure objects."
        )

    exposure_list = list(exposures)

    if not exposure_list:
        raise ValueError(
            "exposures cannot be empty."
        )

    total_exposure = 0.0
    total_expected_loss = 0.0
    total_maximum_loss = 0.0
    unexpected_loss_variance = 0.0

    for exposure in exposure_list:
        if not isinstance(exposure, CreditExposure):
            raise TypeError(
                "Every portfolio item must be a CreditExposure."
            )

        metrics = calculate_credit_risk_metrics(
            probability_of_default=(
                exposure.probability_of_default
            ),
            loss_given_default=exposure.loss_given_default,
            exposure_at_default=exposure.exposure_at_default,
        )

        total_exposure += float(exposure.exposure_at_default)
        total_expected_loss += metrics.expected_loss
        total_maximum_loss += metrics.maximum_loss
        unexpected_loss_variance += (
            metrics.unexpected_loss**2
        )

    return PortfolioCreditRisk(
        exposure_count=len(exposure_list),
        total_exposure=total_exposure,
        expected_loss=total_expected_loss,
        unexpected_loss=sqrt(unexpected_loss_variance),
        maximum_loss=total_maximum_loss,
    )


def main() -> None:
    """Run an illustrative portfolio credit-risk example."""

    exposures = [
        CreditExposure(
            identifier="Corporate Loan A",
            probability_of_default=0.02,
            loss_given_default=0.45,
            exposure_at_default=1_000_000.0,
        ),
        CreditExposure(
            identifier="Corporate Loan B",
            probability_of_default=0.04,
            loss_given_default=0.50,
            exposure_at_default=750_000.0,
        ),
        CreditExposure(
            identifier="Consumer Portfolio",
            probability_of_default=0.06,
            loss_given_default=0.65,
            exposure_at_default=500_000.0,
        ),
    ]

    print("Credit risk modeling engine")
    print("\nIndividual exposures")

    for exposure in exposures:
        metrics = calculate_credit_risk_metrics(
            probability_of_default=(
                exposure.probability_of_default
            ),
            loss_given_default=exposure.loss_given_default,
            exposure_at_default=exposure.exposure_at_default,
        )

        print(f"\n{exposure.identifier}")
        print(
            "Probability of default: "
            f"{exposure.probability_of_default:.2%}"
        )
        print(
            "Loss given default: "
            f"{exposure.loss_given_default:.2%}"
        )
        print(
            "Exposure at default: "
            f"${exposure.exposure_at_default:,.2f}"
        )
        print(
            f"Expected loss: ${metrics.expected_loss:,.2f}"
        )
        print(
            f"Unexpected loss: ${metrics.unexpected_loss:,.2f}"
        )
        print(
            f"Maximum loss: ${metrics.maximum_loss:,.2f}"
        )

    portfolio = calculate_portfolio_credit_risk(exposures)

    print("\nPortfolio summary")
    print(f"Exposures: {portfolio.exposure_count}")
    print(
        f"Total exposure: ${portfolio.total_exposure:,.2f}"
    )
    print(
        f"Expected loss: ${portfolio.expected_loss:,.2f}"
    )
    print(
        f"Unexpected loss: ${portfolio.unexpected_loss:,.2f}"
    )
    print(
        f"Maximum loss: ${portfolio.maximum_loss:,.2f}"
    )


if __name__ == "__main__":
    main()