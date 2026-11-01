"""Tests for correlated credit-default Monte Carlo simulation."""

import numpy as np
import pytest

from src.default_simulation import (
    CreditPortfolio,
    DefaultSimulationResult,
    calculate_analytical_expected_loss,
    create_credit_portfolio,
    simulate_correlated_defaults,
)


@pytest.fixture(scope="module")
def credit_portfolio() -> CreditPortfolio:
    """Create a representative portfolio for simulation tests."""

    exposure_count = 20

    return create_credit_portfolio(
        probabilities_of_default=np.linspace(
            0.01,
            0.10,
            exposure_count,
        ),
        losses_given_default=np.full(
            exposure_count,
            0.45,
        ),
        exposures_at_default=np.linspace(
            100_000.0,
            500_000.0,
            exposure_count,
        ),
        identifiers=[
            f"Borrower_{index + 1}"
            for index in range(exposure_count)
        ],
    )


@pytest.fixture(scope="module")
def simulation_result(
    credit_portfolio: CreditPortfolio,
) -> DefaultSimulationResult:
    """Run one reusable simulation."""

    return simulate_correlated_defaults(
        portfolio=credit_portfolio,
        number_of_simulations=20_000,
        asset_correlation=0.20,
        confidence_level=0.99,
        random_state=42,
    )


def test_create_portfolio_returns_dataclass(
    credit_portfolio: CreditPortfolio,
) -> None:
    """Portfolio creation should return structured data."""

    assert isinstance(credit_portfolio, CreditPortfolio)


def test_portfolio_contains_correct_exposure_count(
    credit_portfolio: CreditPortfolio,
) -> None:
    """Portfolio should report its number of borrowers."""

    assert credit_portfolio.exposure_count == 20
    assert len(credit_portfolio.identifiers) == 20


def test_portfolio_total_exposure(
    credit_portfolio: CreditPortfolio,
) -> None:
    """Total exposure should equal the EAD vector sum."""

    assert credit_portfolio.total_exposure == pytest.approx(
        credit_portfolio.exposures_at_default.sum()
    )


def test_portfolio_arrays_are_read_only(
    credit_portfolio: CreditPortfolio,
) -> None:
    """Validated portfolio parameters should be immutable."""

    assert not credit_portfolio.probabilities_of_default.flags.writeable
    assert not credit_portfolio.losses_given_default.flags.writeable
    assert not credit_portfolio.exposures_at_default.flags.writeable

    with pytest.raises(ValueError):
        credit_portfolio.probabilities_of_default[0] = 0.50


def test_default_identifiers_are_generated() -> None:
    """Identifiers should be generated automatically when omitted."""

    portfolio = create_credit_portfolio(
        probabilities_of_default=[0.01, 0.02],
        losses_given_default=[0.45, 0.50],
        exposures_at_default=[100_000.0, 200_000.0],
    )

    assert portfolio.identifiers == (
        "Exposure_1",
        "Exposure_2",
    )


def test_analytical_expected_loss_matches_formula() -> None:
    """Analytical loss should equal the sum of PD × LGD × EAD."""

    portfolio = create_credit_portfolio(
        probabilities_of_default=[0.02, 0.05],
        losses_given_default=[0.45, 0.60],
        exposures_at_default=[1_000_000.0, 500_000.0],
    )

    result = calculate_analytical_expected_loss(portfolio)

    expected = (
        0.02 * 0.45 * 1_000_000.0
        + 0.05 * 0.60 * 500_000.0
    )

    assert result == pytest.approx(expected)
    assert result == pytest.approx(24_000.0)


def test_simulation_returns_dataclass(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Simulation should return a structured result."""

    assert isinstance(
        simulation_result,
        DefaultSimulationResult,
    )


def test_simulation_preserves_parameters(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Result should report the requested simulation settings."""

    assert simulation_result.number_of_simulations == 20_000
    assert simulation_result.asset_correlation == pytest.approx(0.20)
    assert simulation_result.confidence_level == pytest.approx(0.99)


def test_loss_distribution_has_correct_size(
    simulation_result: DefaultSimulationResult,
) -> None:
    """There should be one loss for every simulation."""

    assert simulation_result.loss_distribution.shape == (20_000,)
    assert (
        simulation_result.default_count_distribution.shape
        == (20_000,)
    )


def test_simulation_distributions_are_read_only(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Stored Monte Carlo results should be immutable."""

    assert not simulation_result.loss_distribution.flags.writeable
    assert (
        not simulation_result
        .default_count_distribution
        .flags.writeable
    )


def test_simulated_loss_is_close_to_analytical_loss(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Monte Carlo mean should converge toward analytical EL."""

    assert simulation_result.simulated_expected_loss == pytest.approx(
        simulation_result.analytical_expected_loss,
        rel=0.10,
    )


def test_average_defaults_close_to_sum_of_pd(
    credit_portfolio: CreditPortfolio,
    simulation_result: DefaultSimulationResult,
) -> None:
    """Average simulated defaults should approach total PD."""

    theoretical_average = float(
        credit_portfolio.probabilities_of_default.sum()
    )

    assert simulation_result.average_default_count == pytest.approx(
        theoretical_average,
        rel=0.10,
    )


def test_expected_shortfall_not_below_value_at_risk(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Expected Shortfall should be at least as large as VaR."""

    assert (
        simulation_result.expected_shortfall
        >= simulation_result.value_at_risk
    )


def test_economic_capital_matches_definition(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Economic capital should equal VaR minus expected loss."""

    expected = max(
        simulation_result.value_at_risk
        - simulation_result.analytical_expected_loss,
        0.0,
    )

    assert simulation_result.economic_capital == pytest.approx(
        expected
    )


def test_maximum_loss_not_below_value_at_risk(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Maximum simulated loss should be at least VaR."""

    assert (
        simulation_result.maximum_simulated_loss
        >= simulation_result.value_at_risk
    )


def test_default_counts_respect_portfolio_size(
    credit_portfolio: CreditPortfolio,
    simulation_result: DefaultSimulationResult,
) -> None:
    """Defaults cannot exceed the number of borrowers."""

    assert simulation_result.maximum_default_count <= (
        credit_portfolio.exposure_count
    )
    assert np.all(
        simulation_result.default_count_distribution >= 0
    )
    assert np.all(
        simulation_result.default_count_distribution
        <= credit_portfolio.exposure_count
    )


def test_simulation_is_reproducible(
    credit_portfolio: CreditPortfolio,
) -> None:
    """The same random seed should reproduce all losses."""

    first = simulate_correlated_defaults(
        portfolio=credit_portfolio,
        number_of_simulations=5_000,
        random_state=123,
    )
    second = simulate_correlated_defaults(
        portfolio=credit_portfolio,
        number_of_simulations=5_000,
        random_state=123,
    )

    assert np.array_equal(
        first.loss_distribution,
        second.loss_distribution,
    )
    assert np.array_equal(
        first.default_count_distribution,
        second.default_count_distribution,
    )


def test_different_seeds_produce_different_results(
    credit_portfolio: CreditPortfolio,
) -> None:
    """Different random seeds should change simulated paths."""

    first = simulate_correlated_defaults(
        portfolio=credit_portfolio,
        number_of_simulations=5_000,
        random_state=1,
    )
    second = simulate_correlated_defaults(
        portfolio=credit_portfolio,
        number_of_simulations=5_000,
        random_state=2,
    )

    assert not np.array_equal(
        first.loss_distribution,
        second.loss_distribution,
    )


def test_higher_correlation_increases_loss_volatility(
    credit_portfolio: CreditPortfolio,
) -> None:
    """Common economic exposure should increase tail concentration."""

    independent = simulate_correlated_defaults(
        portfolio=credit_portfolio,
        number_of_simulations=20_000,
        asset_correlation=0.0,
        random_state=42,
    )
    correlated = simulate_correlated_defaults(
        portfolio=credit_portfolio,
        number_of_simulations=20_000,
        asset_correlation=0.50,
        random_state=42,
    )

    assert (
        correlated.loss_standard_deviation
        > independent.loss_standard_deviation
    )


def test_zero_pd_portfolio_has_no_losses() -> None:
    """Borrowers with zero PD should never default."""

    portfolio = create_credit_portfolio(
        probabilities_of_default=[0.0, 0.0, 0.0],
        losses_given_default=[0.45, 0.50, 0.60],
        exposures_at_default=[
            100_000.0,
            200_000.0,
            300_000.0,
        ],
    )

    result = simulate_correlated_defaults(
        portfolio=portfolio,
        number_of_simulations=1_000,
        random_state=42,
    )

    assert np.all(result.loss_distribution == 0.0)
    assert result.simulated_expected_loss == pytest.approx(0.0)
    assert result.value_at_risk == pytest.approx(0.0)
    assert result.expected_shortfall == pytest.approx(0.0)


def test_certain_default_portfolio_has_fixed_loss() -> None:
    """PD equal to one should produce default in every scenario."""

    portfolio = create_credit_portfolio(
        probabilities_of_default=[1.0, 1.0],
        losses_given_default=[0.50, 0.40],
        exposures_at_default=[100_000.0, 200_000.0],
    )

    result = simulate_correlated_defaults(
        portfolio=portfolio,
        number_of_simulations=1_000,
        random_state=42,
    )

    expected_loss = 130_000.0

    assert np.all(
        result.loss_distribution == expected_loss
    )
    assert result.simulated_expected_loss == pytest.approx(
        expected_loss
    )
    assert result.average_default_count == pytest.approx(2.0)


def test_unequal_vector_lengths_raise_value_error() -> None:
    """PD, LGD, and EAD vectors require equal lengths."""

    with pytest.raises(
        ValueError,
        match="must have equal lengths",
    ):
        create_credit_portfolio(
            probabilities_of_default=[0.01, 0.02],
            losses_given_default=[0.45],
            exposures_at_default=[100_000.0, 200_000.0],
        )


@pytest.mark.parametrize(
    "invalid_pd",
    [-0.01, 1.01],
)
def test_invalid_pd_raises_value_error(
    invalid_pd: float,
) -> None:
    """PD must remain between zero and one."""

    with pytest.raises(
        ValueError,
        match="Probabilities of default must be between 0 and 1",
    ):
        create_credit_portfolio(
            probabilities_of_default=[0.01, invalid_pd],
            losses_given_default=[0.45, 0.50],
            exposures_at_default=[100_000.0, 200_000.0],
        )


@pytest.mark.parametrize(
    "invalid_lgd",
    [-0.01, 1.01],
)
def test_invalid_lgd_raises_value_error(
    invalid_lgd: float,
) -> None:
    """LGD must remain between zero and one."""

    with pytest.raises(
        ValueError,
        match="Losses given default must be between 0 and 1",
    ):
        create_credit_portfolio(
            probabilities_of_default=[0.01, 0.02],
            losses_given_default=[0.45, invalid_lgd],
            exposures_at_default=[100_000.0, 200_000.0],
        )


def test_negative_ead_raises_value_error() -> None:
    """Exposure at default cannot be negative."""

    with pytest.raises(
        ValueError,
        match="Exposures at default cannot be negative",
    ):
        create_credit_portfolio(
            probabilities_of_default=[0.01, 0.02],
            losses_given_default=[0.45, 0.50],
            exposures_at_default=[100_000.0, -200_000.0],
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_non_finite_portfolio_values_raise_value_error(
    invalid_value: float,
) -> None:
    """Portfolio parameters must contain finite values."""

    with pytest.raises(
        ValueError,
        match="must contain only finite values",
    ):
        create_credit_portfolio(
            probabilities_of_default=[0.01, invalid_value],
            losses_given_default=[0.45, 0.50],
            exposures_at_default=[100_000.0, 200_000.0],
        )


def test_non_numeric_portfolio_values_raise_type_error() -> None:
    """Portfolio parameters must be numeric."""

    with pytest.raises(
        TypeError,
        match="must contain numeric values",
    ):
        create_credit_portfolio(
            probabilities_of_default=["low", "high"],
            losses_given_default=[0.45, 0.50],
            exposures_at_default=[100_000.0, 200_000.0],
        )


def test_duplicate_identifiers_raise_value_error() -> None:
    """Every exposure identifier must be unique."""

    with pytest.raises(
        ValueError,
        match="identifiers must be unique",
    ):
        create_credit_portfolio(
            probabilities_of_default=[0.01, 0.02],
            losses_given_default=[0.45, 0.50],
            exposures_at_default=[100_000.0, 200_000.0],
            identifiers=["Loan_A", "Loan_A"],
        )


def test_identifier_count_mismatch_raises_value_error() -> None:
    """Identifier count must match portfolio size."""

    with pytest.raises(
        ValueError,
        match="must match the number of exposures",
    ):
        create_credit_portfolio(
            probabilities_of_default=[0.01, 0.02],
            losses_given_default=[0.45, 0.50],
            exposures_at_default=[100_000.0, 200_000.0],
            identifiers=["Loan_A"],
        )


@pytest.mark.parametrize(
    "invalid_count",
    [0, 1, -100],
)
def test_invalid_simulation_count_raises_value_error(
    credit_portfolio: CreditPortfolio,
    invalid_count: int,
) -> None:
    """At least two Monte Carlo scenarios are required."""

    with pytest.raises(
        ValueError,
        match="must be at least 2",
    ):
        simulate_correlated_defaults(
            portfolio=credit_portfolio,
            number_of_simulations=invalid_count,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [100.5, "1000", True],
)
def test_non_integer_simulation_count_raises_type_error(
    credit_portfolio: CreditPortfolio,
    invalid_count: object,
) -> None:
    """Simulation count must be an integer."""

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        simulate_correlated_defaults(
            portfolio=credit_portfolio,
            number_of_simulations=invalid_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_correlation",
    [-0.01, 1.0, 1.01],
)
def test_invalid_asset_correlation_raises_value_error(
    credit_portfolio: CreditPortfolio,
    invalid_correlation: float,
) -> None:
    """Asset correlation must belong to [0, 1)."""

    with pytest.raises(
        ValueError,
        match="asset_correlation must be in",
    ):
        simulate_correlated_defaults(
            portfolio=credit_portfolio,
            asset_correlation=invalid_correlation,
        )


@pytest.mark.parametrize(
    "invalid_confidence",
    [-0.01, 0.0, 1.0, 1.01],
)
def test_invalid_confidence_level_raises_value_error(
    credit_portfolio: CreditPortfolio,
    invalid_confidence: float,
) -> None:
    """Confidence level must belong to (0, 1)."""

    with pytest.raises(
        ValueError,
        match="confidence_level must be in",
    ):
        simulate_correlated_defaults(
            portfolio=credit_portfolio,
            confidence_level=invalid_confidence,
        )


def test_invalid_random_state_raises_type_error(
    credit_portfolio: CreditPortfolio,
) -> None:
    """Random seed must be an integer or None."""

    with pytest.raises(
        TypeError,
        match="random_state must be an integer or None",
    ):
        simulate_correlated_defaults(
            portfolio=credit_portfolio,
            random_state=42.5,  # type: ignore[arg-type]
        )


def test_invalid_portfolio_raises_type_error() -> None:
    """Simulation requires a validated CreditPortfolio."""

    with pytest.raises(
        TypeError,
        match="portfolio must be a CreditPortfolio",
    ):
        simulate_correlated_defaults(
            portfolio="invalid",  # type: ignore[arg-type]
        )