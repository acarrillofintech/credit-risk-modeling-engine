"""Tests for expected and unexpected credit-loss calculations."""

from dataclasses import FrozenInstanceError
from math import sqrt

import pytest

from src.expected_loss import (
    CreditExposure,
    CreditRiskMetrics,
    PortfolioCreditRisk,
    calculate_credit_risk_metrics,
    calculate_expected_loss,
    calculate_maximum_loss,
    calculate_portfolio_credit_risk,
    calculate_unexpected_loss,
)


@pytest.fixture
def sample_exposures() -> list[CreditExposure]:
    """Create a representative credit portfolio."""

    return [
        CreditExposure(
            identifier="Loan A",
            probability_of_default=0.02,
            loss_given_default=0.45,
            exposure_at_default=1_000_000.0,
        ),
        CreditExposure(
            identifier="Loan B",
            probability_of_default=0.04,
            loss_given_default=0.50,
            exposure_at_default=750_000.0,
        ),
        CreditExposure(
            identifier="Loan C",
            probability_of_default=0.06,
            loss_given_default=0.65,
            exposure_at_default=500_000.0,
        ),
    ]


def test_expected_loss_matches_known_value() -> None:
    """Expected loss should equal PD multiplied by LGD and EAD."""

    result = calculate_expected_loss(
        probability_of_default=0.02,
        loss_given_default=0.45,
        exposure_at_default=1_000_000.0,
    )

    assert result == pytest.approx(9_000.0)


@pytest.mark.parametrize(
    ("probability_of_default", "expected_loss"),
    [
        (0.0, 0.0),
        (0.25, 112_500.0),
        (1.0, 450_000.0),
    ],
)
def test_expected_loss_probability_boundaries(
    probability_of_default: float,
    expected_loss: float,
) -> None:
    """Expected loss should support valid probability boundaries."""

    result = calculate_expected_loss(
        probability_of_default=probability_of_default,
        loss_given_default=0.45,
        exposure_at_default=1_000_000.0,
    )

    assert result == pytest.approx(expected_loss)


def test_unexpected_loss_matches_bernoulli_model() -> None:
    """Unexpected loss should match the Bernoulli default formula."""

    result = calculate_unexpected_loss(
        probability_of_default=0.02,
        loss_given_default=0.45,
        exposure_at_default=1_000_000.0,
    )

    expected = 0.45 * 1_000_000.0 * sqrt(
        0.02 * (1.0 - 0.02)
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "probability_of_default",
    [0.0, 1.0],
)
def test_unexpected_loss_is_zero_at_probability_boundaries(
    probability_of_default: float,
) -> None:
    """There is no default uncertainty at PD equal to zero or one."""

    result = calculate_unexpected_loss(
        probability_of_default=probability_of_default,
        loss_given_default=0.45,
        exposure_at_default=1_000_000.0,
    )

    assert result == pytest.approx(0.0)


def test_maximum_loss_matches_lgd_times_ead() -> None:
    """Maximum loss should equal LGD multiplied by EAD."""

    result = calculate_maximum_loss(
        loss_given_default=0.45,
        exposure_at_default=1_000_000.0,
    )

    assert result == pytest.approx(450_000.0)


def test_credit_risk_metrics_returns_dataclass() -> None:
    """Combined calculation should return CreditRiskMetrics."""

    result = calculate_credit_risk_metrics(
        probability_of_default=0.02,
        loss_given_default=0.45,
        exposure_at_default=1_000_000.0,
    )

    assert isinstance(result, CreditRiskMetrics)
    assert result.expected_loss == pytest.approx(9_000.0)
    assert result.unexpected_loss == pytest.approx(
        450_000.0 * sqrt(0.02 * 0.98)
    )
    assert result.maximum_loss == pytest.approx(450_000.0)


def test_credit_exposure_is_immutable() -> None:
    """Credit exposures should not change after creation."""

    exposure = CreditExposure(
        identifier="Loan A",
        probability_of_default=0.02,
        loss_given_default=0.45,
        exposure_at_default=1_000_000.0,
    )

    with pytest.raises(FrozenInstanceError):
        exposure.probability_of_default = 0.50  # type: ignore[misc]


def test_portfolio_returns_dataclass(
    sample_exposures: list[CreditExposure],
) -> None:
    """Portfolio calculation should return PortfolioCreditRisk."""

    result = calculate_portfolio_credit_risk(sample_exposures)

    assert isinstance(result, PortfolioCreditRisk)


def test_portfolio_exposure_count(
    sample_exposures: list[CreditExposure],
) -> None:
    """Portfolio should report the correct number of exposures."""

    result = calculate_portfolio_credit_risk(sample_exposures)

    assert result.exposure_count == 3


def test_portfolio_total_exposure(
    sample_exposures: list[CreditExposure],
) -> None:
    """Portfolio exposure should equal the sum of all EAD values."""

    result = calculate_portfolio_credit_risk(sample_exposures)

    assert result.total_exposure == pytest.approx(2_250_000.0)


def test_portfolio_expected_loss(
    sample_exposures: list[CreditExposure],
) -> None:
    """Portfolio expected loss should sum individual losses."""

    result = calculate_portfolio_credit_risk(sample_exposures)

    expected = (
        0.02 * 0.45 * 1_000_000.0
        + 0.04 * 0.50 * 750_000.0
        + 0.06 * 0.65 * 500_000.0
    )

    assert result.expected_loss == pytest.approx(expected)
    assert result.expected_loss == pytest.approx(43_500.0)


def test_portfolio_maximum_loss(
    sample_exposures: list[CreditExposure],
) -> None:
    """Maximum portfolio loss should sum individual maximum losses."""

    result = calculate_portfolio_credit_risk(sample_exposures)

    assert result.maximum_loss == pytest.approx(1_150_000.0)


def test_portfolio_unexpected_loss_uses_independence(
    sample_exposures: list[CreditExposure],
) -> None:
    """Independent unexpected losses should combine quadratically."""

    individual_unexpected_losses = [
        calculate_unexpected_loss(
            probability_of_default=exposure.probability_of_default,
            loss_given_default=exposure.loss_given_default,
            exposure_at_default=exposure.exposure_at_default,
        )
        for exposure in sample_exposures
    ]

    expected = sqrt(
        sum(
            unexpected_loss**2
            for unexpected_loss in individual_unexpected_losses
        )
    )

    result = calculate_portfolio_credit_risk(sample_exposures)

    assert result.unexpected_loss == pytest.approx(expected)


def test_diversification_reduces_aggregated_uncertainty(
    sample_exposures: list[CreditExposure],
) -> None:
    """Independent portfolio UL should be below the direct sum."""

    individual_total = sum(
        calculate_unexpected_loss(
            probability_of_default=exposure.probability_of_default,
            loss_given_default=exposure.loss_given_default,
            exposure_at_default=exposure.exposure_at_default,
        )
        for exposure in sample_exposures
    )

    portfolio = calculate_portfolio_credit_risk(
        sample_exposures
    )

    assert portfolio.unexpected_loss < individual_total


def test_portfolio_accepts_tuple(
    sample_exposures: list[CreditExposure],
) -> None:
    """A tuple of credit exposures should be accepted."""

    result = calculate_portfolio_credit_risk(
        tuple(sample_exposures)
    )

    assert result.exposure_count == 3


def test_empty_portfolio_raises_value_error() -> None:
    """An empty portfolio should not produce risk metrics."""

    with pytest.raises(
        ValueError,
        match="exposures cannot be empty",
    ):
        calculate_portfolio_credit_risk([])


def test_text_portfolio_raises_type_error() -> None:
    """Text should not be interpreted as a portfolio."""

    with pytest.raises(
        TypeError,
        match="exposures must be a sequence",
    ):
        calculate_portfolio_credit_risk("Loan A")  # type: ignore[arg-type]


def test_invalid_portfolio_item_raises_type_error() -> None:
    """Every portfolio item must be a CreditExposure."""

    with pytest.raises(
        TypeError,
        match="Every portfolio item must be a CreditExposure",
    ):
        calculate_portfolio_credit_risk(
            [100]  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    "probability_of_default",
    [-0.01, 1.01],
)
def test_invalid_pd_raises_value_error(
    probability_of_default: float,
) -> None:
    """PD values outside zero and one should be rejected."""

    with pytest.raises(
        ValueError,
        match="probability_of_default must be between 0 and 1",
    ):
        calculate_expected_loss(
            probability_of_default=probability_of_default,
            loss_given_default=0.45,
            exposure_at_default=1_000_000.0,
        )


@pytest.mark.parametrize(
    "loss_given_default",
    [-0.01, 1.01],
)
def test_invalid_lgd_raises_value_error(
    loss_given_default: float,
) -> None:
    """LGD values outside zero and one should be rejected."""

    with pytest.raises(
        ValueError,
        match="loss_given_default must be between 0 and 1",
    ):
        calculate_expected_loss(
            probability_of_default=0.02,
            loss_given_default=loss_given_default,
            exposure_at_default=1_000_000.0,
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_probability_raises_value_error(
    invalid_probability: float,
) -> None:
    """Probabilities must be finite numbers."""

    with pytest.raises(
        ValueError,
        match="probability_of_default must be finite",
    ):
        calculate_expected_loss(
            probability_of_default=invalid_probability,
            loss_given_default=0.45,
            exposure_at_default=1_000_000.0,
        )


@pytest.mark.parametrize(
    "invalid_probability",
    ["0.02", None, True],
)
def test_non_numeric_probability_raises_type_error(
    invalid_probability: object,
) -> None:
    """Probabilities must be real numeric values."""

    with pytest.raises(
        TypeError,
        match="probability_of_default must be a real number",
    ):
        calculate_expected_loss(
            probability_of_default=invalid_probability,  # type: ignore[arg-type]
            loss_given_default=0.45,
            exposure_at_default=1_000_000.0,
        )


def test_negative_exposure_raises_value_error() -> None:
    """EAD cannot be negative."""

    with pytest.raises(
        ValueError,
        match="exposure_at_default cannot be negative",
    ):
        calculate_expected_loss(
            probability_of_default=0.02,
            loss_given_default=0.45,
            exposure_at_default=-1_000.0,
        )


@pytest.mark.parametrize(
    "invalid_exposure",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_exposure_raises_value_error(
    invalid_exposure: float,
) -> None:
    """EAD must contain a finite monetary value."""

    with pytest.raises(
        ValueError,
        match="exposure_at_default must be finite",
    ):
        calculate_expected_loss(
            probability_of_default=0.02,
            loss_given_default=0.45,
            exposure_at_default=invalid_exposure,
        )


@pytest.mark.parametrize(
    "invalid_exposure",
    ["1000000", None, True],
)
def test_non_numeric_exposure_raises_type_error(
    invalid_exposure: object,
) -> None:
    """EAD must be a real numeric value."""

    with pytest.raises(
        TypeError,
        match="exposure_at_default must be a real number",
    ):
        calculate_expected_loss(
            probability_of_default=0.02,
            loss_given_default=0.45,
            exposure_at_default=invalid_exposure,  # type: ignore[arg-type]
        )