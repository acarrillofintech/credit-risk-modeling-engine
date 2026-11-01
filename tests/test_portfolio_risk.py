"""Tests for portfolio concentration and credit stress testing."""

import numpy as np
import pytest

from src.default_simulation import (
    CreditPortfolio,
    create_credit_portfolio,
)
from src.portfolio_risk import (
    ExposureRiskContribution,
    PortfolioRiskAnalysis,
    SectorRiskSummary,
    StressScenario,
    StressTestResult,
    analyze_portfolio_risk,
    apply_stress_scenario,
    run_stress_scenarios,
    run_stress_test,
)


@pytest.fixture
def portfolio() -> CreditPortfolio:
    """Create a representative four-borrower portfolio."""

    return create_credit_portfolio(
        probabilities_of_default=[
            0.01,
            0.02,
            0.05,
            0.10,
        ],
        losses_given_default=[
            0.40,
            0.45,
            0.50,
            0.60,
        ],
        exposures_at_default=[
            1_000_000.0,
            750_000.0,
            500_000.0,
            250_000.0,
        ],
        identifiers=[
            "Corporate_A",
            "Corporate_B",
            "Retail_A",
            "Retail_B",
        ],
    )


@pytest.fixture
def sectors() -> list[str]:
    """Create sector labels for the sample portfolio."""

    return [
        "Corporate",
        "Corporate",
        "Retail",
        "Retail",
    ]


@pytest.fixture
def analysis(
    portfolio: CreditPortfolio,
    sectors: list[str],
) -> PortfolioRiskAnalysis:
    """Create a reusable portfolio-risk analysis."""

    return analyze_portfolio_risk(
        portfolio=portfolio,
        sectors=sectors,
    )


def test_analysis_returns_dataclass(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Analysis should return structured output."""

    assert isinstance(analysis, PortfolioRiskAnalysis)


def test_analysis_reports_exposure_count(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Exposure count should match the portfolio."""

    assert analysis.exposure_count == 4


def test_analysis_reports_total_exposure(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Total portfolio EAD should be correct."""

    assert analysis.total_exposure == pytest.approx(
        2_500_000.0
    )


def test_expected_loss_matches_known_value(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Portfolio expected loss should equal sum of PD × LGD × EAD."""

    assert analysis.expected_loss == pytest.approx(
        38_250.0
    )


def test_weighted_average_pd(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """PD should be weighted using EAD shares."""

    assert analysis.weighted_average_pd == pytest.approx(
        0.03
    )


def test_weighted_average_lgd(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """LGD should be weighted using EAD shares."""

    assert analysis.weighted_average_lgd == pytest.approx(
        0.455
    )


def test_largest_exposure_share(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Largest borrower should represent forty percent."""

    assert analysis.largest_exposure_share == pytest.approx(
        0.40
    )


def test_hhi_matches_known_value(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """HHI should equal the sum of squared exposure shares."""

    expected_hhi = (
        0.40**2
        + 0.30**2
        + 0.20**2
        + 0.10**2
    )

    assert (
        analysis.herfindahl_hirschman_index
        == pytest.approx(expected_hhi)
    )
    assert expected_hhi == pytest.approx(0.30)


def test_normalized_concentration_index(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Normalized HHI should account for portfolio size."""

    expected = (0.30 - 0.25) / (1.0 - 0.25)

    assert (
        analysis.normalized_concentration_index
        == pytest.approx(expected)
    )


def test_exposure_contributions_are_dataclasses(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Every borrower contribution should be structured."""

    assert len(analysis.exposure_contributions) == 4
    assert all(
        isinstance(item, ExposureRiskContribution)
        for item in analysis.exposure_contributions
    )


def test_exposure_shares_sum_to_one(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Borrower EAD shares should form a complete portfolio."""

    total = sum(
        item.exposure_share
        for item in analysis.exposure_contributions
    )

    assert total == pytest.approx(1.0)


def test_expected_loss_shares_sum_to_one(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Expected-loss contributions should sum to one."""

    total = sum(
        item.expected_loss_share
        for item in analysis.exposure_contributions
    )

    assert total == pytest.approx(1.0)


def test_known_borrower_contribution(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Corporate A should have the correct risk contribution."""

    contribution = analysis.exposure_contributions[0]

    assert contribution.identifier == "Corporate_A"
    assert contribution.exposure_at_default == pytest.approx(
        1_000_000.0
    )
    assert contribution.expected_loss == pytest.approx(
        4_000.0
    )
    assert contribution.exposure_share == pytest.approx(
        0.40
    )
    assert contribution.expected_loss_share == pytest.approx(
        4_000.0 / 38_250.0
    )


def test_sector_summaries_are_dataclasses(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Every sector summary should be structured."""

    assert len(analysis.sector_summaries) == 2
    assert all(
        isinstance(item, SectorRiskSummary)
        for item in analysis.sector_summaries
    )


def test_corporate_sector_summary(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Corporate sector aggregation should be correct."""

    corporate = next(
        sector
        for sector in analysis.sector_summaries
        if sector.sector == "Corporate"
    )

    assert corporate.exposure_count == 2
    assert corporate.total_exposure == pytest.approx(
        1_750_000.0
    )
    assert corporate.expected_loss == pytest.approx(
        10_750.0
    )
    assert corporate.exposure_share == pytest.approx(
        0.70
    )


def test_sector_exposure_shares_sum_to_one(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Sector exposure shares should sum to one."""

    assert sum(
        sector.exposure_share
        for sector in analysis.sector_summaries
    ) == pytest.approx(1.0)


def test_sector_loss_shares_sum_to_one(
    analysis: PortfolioRiskAnalysis,
) -> None:
    """Sector expected-loss shares should sum to one."""

    assert sum(
        sector.expected_loss_share
        for sector in analysis.sector_summaries
    ) == pytest.approx(1.0)


def test_missing_sectors_use_unclassified(
    portfolio: CreditPortfolio,
) -> None:
    """Omitted sectors should be grouped as Unclassified."""

    result = analyze_portfolio_risk(portfolio)

    assert len(result.sector_summaries) == 1
    assert (
        result.sector_summaries[0].sector
        == "Unclassified"
    )
    assert result.sector_summaries[0].exposure_count == 4


def test_equal_portfolio_has_minimum_hhi() -> None:
    """Equal exposures should produce HHI equal to one over N."""

    equal_portfolio = create_credit_portfolio(
        probabilities_of_default=[0.02] * 4,
        losses_given_default=[0.45] * 4,
        exposures_at_default=[100_000.0] * 4,
    )

    result = analyze_portfolio_risk(equal_portfolio)

    assert result.herfindahl_hirschman_index == pytest.approx(
        0.25
    )
    assert result.normalized_concentration_index == pytest.approx(
        0.0
    )


def test_single_exposure_is_fully_concentrated() -> None:
    """A one-borrower portfolio has maximum concentration."""

    single_portfolio = create_credit_portfolio(
        probabilities_of_default=[0.02],
        losses_given_default=[0.45],
        exposures_at_default=[1_000_000.0],
    )

    result = analyze_portfolio_risk(single_portfolio)

    assert result.herfindahl_hirschman_index == pytest.approx(
        1.0
    )
    assert result.normalized_concentration_index == pytest.approx(
        1.0
    )


def test_zero_expected_loss_has_zero_contributions() -> None:
    """Zero-PD borrowers should have zero loss shares."""

    zero_loss_portfolio = create_credit_portfolio(
        probabilities_of_default=[0.0, 0.0],
        losses_given_default=[0.45, 0.50],
        exposures_at_default=[100_000.0, 200_000.0],
    )

    result = analyze_portfolio_risk(zero_loss_portfolio)

    assert result.expected_loss == pytest.approx(0.0)
    assert all(
        contribution.expected_loss_share == 0.0
        for contribution in result.exposure_contributions
    )


def test_apply_stress_scenario_changes_parameters(
    portfolio: CreditPortfolio,
) -> None:
    """Stress multipliers should modify PD, LGD, and EAD."""

    scenario = StressScenario(
        name="Recession",
        pd_multiplier=2.0,
        lgd_multiplier=1.10,
        ead_multiplier=1.05,
    )

    stressed = apply_stress_scenario(
        portfolio=portfolio,
        scenario=scenario,
    )

    assert np.allclose(
        stressed.probabilities_of_default,
        portfolio.probabilities_of_default * 2.0,
    )
    assert np.allclose(
        stressed.losses_given_default,
        portfolio.losses_given_default * 1.10,
    )
    assert np.allclose(
        stressed.exposures_at_default,
        portfolio.exposures_at_default * 1.05,
    )


def test_stressed_probabilities_are_capped_at_one() -> None:
    """Stress cannot produce PD or LGD above one."""

    stressed_portfolio = create_credit_portfolio(
        probabilities_of_default=[0.60, 0.80],
        losses_given_default=[0.70, 0.90],
        exposures_at_default=[100_000.0, 200_000.0],
    )

    scenario = StressScenario(
        name="Extreme stress",
        pd_multiplier=3.0,
        lgd_multiplier=2.0,
    )

    result = apply_stress_scenario(
        portfolio=stressed_portfolio,
        scenario=scenario,
    )

    assert np.all(
        result.probabilities_of_default <= 1.0
    )
    assert np.all(result.losses_given_default <= 1.0)
    assert np.allclose(
        result.probabilities_of_default,
        [1.0, 1.0],
    )
    assert np.allclose(
        result.losses_given_default,
        [1.0, 1.0],
    )


def test_run_stress_test_returns_dataclass(
    portfolio: CreditPortfolio,
) -> None:
    """Stress test should return structured results."""

    result = run_stress_test(
        portfolio=portfolio,
        scenario=StressScenario(
            name="Recession",
            pd_multiplier=2.0,
            lgd_multiplier=1.10,
            ead_multiplier=1.05,
        ),
    )

    assert isinstance(result, StressTestResult)


def test_stress_test_expected_loss_increase(
    portfolio: CreditPortfolio,
) -> None:
    """Expected loss should scale under an uncapped scenario."""

    scenario = StressScenario(
        name="Recession",
        pd_multiplier=2.0,
        lgd_multiplier=1.10,
        ead_multiplier=1.05,
    )

    result = run_stress_test(
        portfolio=portfolio,
        scenario=scenario,
    )

    expected_stressed_loss = 38_250.0 * 2.0 * 1.10 * 1.05

    assert result.base_expected_loss == pytest.approx(
        38_250.0
    )
    assert result.stressed_expected_loss == pytest.approx(
        expected_stressed_loss
    )
    assert result.absolute_increase == pytest.approx(
        expected_stressed_loss - 38_250.0
    )
    assert result.percentage_increase == pytest.approx(
        2.0 * 1.10 * 1.05 - 1.0
    )


def test_multiple_stress_scenarios(
    portfolio: CreditPortfolio,
) -> None:
    """Multiple scenarios should produce multiple results."""

    scenarios = [
        StressScenario(
            name="Mild",
            pd_multiplier=1.25,
        ),
        StressScenario(
            name="Severe",
            pd_multiplier=2.50,
            lgd_multiplier=1.25,
        ),
    ]

    results = run_stress_scenarios(
        portfolio=portfolio,
        scenarios=scenarios,
    )

    assert len(results) == 2
    assert results[0].scenario.name == "Mild"
    assert results[1].scenario.name == "Severe"
    assert (
        results[1].stressed_expected_loss
        > results[0].stressed_expected_loss
    )


def test_sector_count_mismatch_raises_value_error(
    portfolio: CreditPortfolio,
) -> None:
    """There must be one sector per exposure."""

    with pytest.raises(
        ValueError,
        match="sectors must match the number of exposures",
    ):
        analyze_portfolio_risk(
            portfolio=portfolio,
            sectors=["Corporate", "Retail"],
        )


def test_sector_string_raises_type_error(
    portfolio: CreditPortfolio,
) -> None:
    """A string should not be interpreted as sector sequence."""

    with pytest.raises(
        TypeError,
        match="sectors must be a sequence of strings",
    ):
        analyze_portfolio_risk(
            portfolio=portfolio,
            sectors="Corporate",  # type: ignore[arg-type]
        )


def test_blank_sector_raises_value_error(
    portfolio: CreditPortfolio,
) -> None:
    """Sector names cannot be blank."""

    with pytest.raises(
        ValueError,
        match="Every sector must be a non-empty string",
    ):
        analyze_portfolio_risk(
            portfolio=portfolio,
            sectors=[
                "Corporate",
                "Corporate",
                "",
                "Retail",
            ],
        )


def test_zero_total_exposure_raises_value_error() -> None:
    """Concentration cannot be calculated without exposure."""

    zero_portfolio = create_credit_portfolio(
        probabilities_of_default=[0.01, 0.02],
        losses_given_default=[0.45, 0.50],
        exposures_at_default=[0.0, 0.0],
    )

    with pytest.raises(
        ValueError,
        match="total exposure must be positive",
    ):
        analyze_portfolio_risk(zero_portfolio)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("pd_multiplier", -0.1),
        ("lgd_multiplier", -0.1),
        ("ead_multiplier", -0.1),
    ],
)
def test_negative_multiplier_raises_value_error(
    portfolio: CreditPortfolio,
    field_name: str,
    invalid_value: float,
) -> None:
    """Stress multipliers cannot be negative."""

    parameters = {
        "name": "Invalid",
        "pd_multiplier": 1.0,
        "lgd_multiplier": 1.0,
        "ead_multiplier": 1.0,
    }
    parameters[field_name] = invalid_value

    scenario = StressScenario(**parameters)

    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be negative",
    ):
        apply_stress_scenario(
            portfolio=portfolio,
            scenario=scenario,
        )


def test_blank_scenario_name_raises_value_error(
    portfolio: CreditPortfolio,
) -> None:
    """Stress scenario requires a descriptive name."""

    scenario = StressScenario(
        name="",
        pd_multiplier=2.0,
    )

    with pytest.raises(
        ValueError,
        match="scenario name must be a non-empty string",
    ):
        apply_stress_scenario(
            portfolio=portfolio,
            scenario=scenario,
        )


def test_invalid_scenario_type_raises_type_error(
    portfolio: CreditPortfolio,
) -> None:
    """Stress functions require a StressScenario."""

    with pytest.raises(
        TypeError,
        match="scenario must be a StressScenario",
    ):
        apply_stress_scenario(
            portfolio=portfolio,
            scenario="Recession",  # type: ignore[arg-type]
        )


def test_empty_scenario_collection_raises_value_error(
    portfolio: CreditPortfolio,
) -> None:
    """At least one stress scenario is required."""

    with pytest.raises(
        ValueError,
        match="scenarios cannot be empty",
    ):
        run_stress_scenarios(
            portfolio=portfolio,
            scenarios=[],
        )


def test_invalid_portfolio_raises_type_error() -> None:
    """Portfolio analysis requires a CreditPortfolio."""

    with pytest.raises(
        TypeError,
        match="portfolio must be a CreditPortfolio",
    ):
        analyze_portfolio_risk(
            portfolio="invalid",  # type: ignore[arg-type]
        )