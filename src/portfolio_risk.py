"""Credit portfolio concentration and stress testing.

This module calculates:

- exposure concentration;
- expected-loss contributions;
- sector-level credit risk;
- weighted-average PD and LGD;
- stressed losses under adverse economic scenarios.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Sequence

import numpy as np

from src.default_simulation import (
    CreditPortfolio,
    calculate_analytical_expected_loss,
    create_credit_portfolio,
)


@dataclass(frozen=True)
class ExposureRiskContribution:
    """Risk contribution from one credit exposure."""

    identifier: str
    exposure_at_default: float
    expected_loss: float
    exposure_share: float
    expected_loss_share: float


@dataclass(frozen=True)
class SectorRiskSummary:
    """Aggregated credit risk for one economic sector."""

    sector: str
    exposure_count: int
    total_exposure: float
    expected_loss: float
    exposure_share: float
    expected_loss_share: float


@dataclass(frozen=True)
class PortfolioRiskAnalysis:
    """Concentration and expected-loss analysis."""

    exposure_count: int
    total_exposure: float
    expected_loss: float
    weighted_average_pd: float
    weighted_average_lgd: float
    largest_exposure_share: float
    herfindahl_hirschman_index: float
    normalized_concentration_index: float
    exposure_contributions: tuple[
        ExposureRiskContribution,
        ...,
    ]
    sector_summaries: tuple[SectorRiskSummary, ...]


@dataclass(frozen=True)
class StressScenario:
    """Multiplicative credit-risk stress scenario."""

    name: str
    pd_multiplier: float = 1.0
    lgd_multiplier: float = 1.0
    ead_multiplier: float = 1.0


@dataclass(frozen=True)
class StressTestResult:
    """Expected-loss impact from one stress scenario."""

    scenario: StressScenario
    base_expected_loss: float
    stressed_expected_loss: float
    absolute_increase: float
    percentage_increase: float
    stressed_portfolio: CreditPortfolio


def _validate_sectors(
    sectors: Sequence[str] | None,
    exposure_count: int,
) -> tuple[str, ...]:
    """Validate or generate sector labels."""

    if sectors is None:
        return tuple(
            "Unclassified"
            for _ in range(exposure_count)
        )

    if isinstance(sectors, (str, bytes)):
        raise TypeError(
            "sectors must be a sequence of strings."
        )

    normalized_sectors = tuple(sectors)

    if len(normalized_sectors) != exposure_count:
        raise ValueError(
            "sectors must match the number of exposures."
        )

    if any(
        not isinstance(sector, str) or not sector.strip()
        for sector in normalized_sectors
    ):
        raise ValueError(
            "Every sector must be a non-empty string."
        )

    return normalized_sectors


def _validate_multiplier(
    multiplier: float,
    parameter_name: str,
) -> float:
    """Validate a non-negative stress multiplier."""

    if isinstance(multiplier, bool) or not isinstance(
        multiplier,
        Real,
    ):
        raise TypeError(
            f"{parameter_name} must be a real number."
        )

    normalized_multiplier = float(multiplier)

    if not isfinite(normalized_multiplier):
        raise ValueError(
            f"{parameter_name} must be finite."
        )

    if normalized_multiplier < 0.0:
        raise ValueError(
            f"{parameter_name} cannot be negative."
        )

    return normalized_multiplier


def _validate_scenario(
    scenario: StressScenario,
) -> tuple[float, float, float]:
    """Validate a stress scenario and return its multipliers."""

    if not isinstance(scenario, StressScenario):
        raise TypeError(
            "scenario must be a StressScenario."
        )

    if not isinstance(scenario.name, str) or not scenario.name.strip():
        raise ValueError(
            "scenario name must be a non-empty string."
        )

    pd_multiplier = _validate_multiplier(
        scenario.pd_multiplier,
        "pd_multiplier",
    )
    lgd_multiplier = _validate_multiplier(
        scenario.lgd_multiplier,
        "lgd_multiplier",
    )
    ead_multiplier = _validate_multiplier(
        scenario.ead_multiplier,
        "ead_multiplier",
    )

    return (
        pd_multiplier,
        lgd_multiplier,
        ead_multiplier,
    )


def analyze_portfolio_risk(
    portfolio: CreditPortfolio,
    sectors: Sequence[str] | None = None,
) -> PortfolioRiskAnalysis:
    """Analyze credit concentration and loss contributions."""

    if not isinstance(portfolio, CreditPortfolio):
        raise TypeError(
            "portfolio must be a CreditPortfolio."
        )

    if portfolio.total_exposure <= 0.0:
        raise ValueError(
            "portfolio total exposure must be positive."
        )

    normalized_sectors = _validate_sectors(
        sectors=sectors,
        exposure_count=portfolio.exposure_count,
    )

    individual_expected_losses = (
        portfolio.probabilities_of_default
        * portfolio.losses_given_default
        * portfolio.exposures_at_default
    )

    total_expected_loss = float(
        individual_expected_losses.sum()
    )

    exposure_shares = (
        portfolio.exposures_at_default
        / portfolio.total_exposure
    )

    if total_expected_loss > 0.0:
        expected_loss_shares = (
            individual_expected_losses
            / total_expected_loss
        )
    else:
        expected_loss_shares = np.zeros(
            portfolio.exposure_count,
            dtype=float,
        )

    weighted_average_pd = float(
        np.sum(
            portfolio.probabilities_of_default
            * exposure_shares
        )
    )

    weighted_average_lgd = float(
        np.sum(
            portfolio.losses_given_default
            * exposure_shares
        )
    )

    concentration_index = float(
        np.sum(exposure_shares**2)
    )

    if portfolio.exposure_count == 1:
        normalized_concentration = 1.0
    else:
        minimum_concentration = (
            1.0 / portfolio.exposure_count
        )
        normalized_concentration = (
            concentration_index - minimum_concentration
        ) / (1.0 - minimum_concentration)

    contributions = tuple(
        ExposureRiskContribution(
            identifier=identifier,
            exposure_at_default=float(ead),
            expected_loss=float(expected_loss),
            exposure_share=float(exposure_share),
            expected_loss_share=float(expected_loss_share),
        )
        for (
            identifier,
            ead,
            expected_loss,
            exposure_share,
            expected_loss_share,
        ) in zip(
            portfolio.identifiers,
            portfolio.exposures_at_default,
            individual_expected_losses,
            exposure_shares,
            expected_loss_shares,
            strict=True,
        )
    )

    sector_summaries: list[SectorRiskSummary] = []

    for sector in sorted(set(normalized_sectors)):
        sector_mask = np.array(
            [
                borrower_sector == sector
                for borrower_sector in normalized_sectors
            ],
            dtype=bool,
        )

        sector_exposure = float(
            portfolio.exposures_at_default[
                sector_mask
            ].sum()
        )
        sector_expected_loss = float(
            individual_expected_losses[
                sector_mask
            ].sum()
        )

        sector_summaries.append(
            SectorRiskSummary(
                sector=sector,
                exposure_count=int(
                    np.count_nonzero(sector_mask)
                ),
                total_exposure=sector_exposure,
                expected_loss=sector_expected_loss,
                exposure_share=(
                    sector_exposure
                    / portfolio.total_exposure
                ),
                expected_loss_share=(
                    sector_expected_loss
                    / total_expected_loss
                    if total_expected_loss > 0.0
                    else 0.0
                ),
            )
        )

    return PortfolioRiskAnalysis(
        exposure_count=portfolio.exposure_count,
        total_exposure=portfolio.total_exposure,
        expected_loss=total_expected_loss,
        weighted_average_pd=weighted_average_pd,
        weighted_average_lgd=weighted_average_lgd,
        largest_exposure_share=float(
            np.max(exposure_shares)
        ),
        herfindahl_hirschman_index=concentration_index,
        normalized_concentration_index=float(
            normalized_concentration
        ),
        exposure_contributions=contributions,
        sector_summaries=tuple(sector_summaries),
    )


def apply_stress_scenario(
    portfolio: CreditPortfolio,
    scenario: StressScenario,
) -> CreditPortfolio:
    """Apply adverse multipliers to PD, LGD, and EAD."""

    if not isinstance(portfolio, CreditPortfolio):
        raise TypeError(
            "portfolio must be a CreditPortfolio."
        )

    (
        pd_multiplier,
        lgd_multiplier,
        ead_multiplier,
    ) = _validate_scenario(scenario)

    stressed_probabilities = np.minimum(
        portfolio.probabilities_of_default
        * pd_multiplier,
        1.0,
    )
    stressed_losses = np.minimum(
        portfolio.losses_given_default
        * lgd_multiplier,
        1.0,
    )
    stressed_exposures = (
        portfolio.exposures_at_default
        * ead_multiplier
    )

    return create_credit_portfolio(
        probabilities_of_default=stressed_probabilities,
        losses_given_default=stressed_losses,
        exposures_at_default=stressed_exposures,
        identifiers=portfolio.identifiers,
    )


def run_stress_test(
    portfolio: CreditPortfolio,
    scenario: StressScenario,
) -> StressTestResult:
    """Calculate expected-loss impact for one scenario."""

    if not isinstance(portfolio, CreditPortfolio):
        raise TypeError(
            "portfolio must be a CreditPortfolio."
        )

    stressed_portfolio = apply_stress_scenario(
        portfolio=portfolio,
        scenario=scenario,
    )

    base_expected_loss = (
        calculate_analytical_expected_loss(portfolio)
    )
    stressed_expected_loss = (
        calculate_analytical_expected_loss(
            stressed_portfolio
        )
    )

    absolute_increase = (
        stressed_expected_loss
        - base_expected_loss
    )

    if base_expected_loss > 0.0:
        percentage_increase = (
            absolute_increase / base_expected_loss
        )
    elif stressed_expected_loss > 0.0:
        percentage_increase = float("inf")
    else:
        percentage_increase = 0.0

    return StressTestResult(
        scenario=scenario,
        base_expected_loss=base_expected_loss,
        stressed_expected_loss=stressed_expected_loss,
        absolute_increase=absolute_increase,
        percentage_increase=percentage_increase,
        stressed_portfolio=stressed_portfolio,
    )


def run_stress_scenarios(
    portfolio: CreditPortfolio,
    scenarios: Sequence[StressScenario],
) -> tuple[StressTestResult, ...]:
    """Run multiple credit-risk stress scenarios."""

    if not isinstance(portfolio, CreditPortfolio):
        raise TypeError(
            "portfolio must be a CreditPortfolio."
        )

    if isinstance(scenarios, (str, bytes)):
        raise TypeError(
            "scenarios must be a sequence of StressScenario objects."
        )

    normalized_scenarios = tuple(scenarios)

    if not normalized_scenarios:
        raise ValueError(
            "scenarios cannot be empty."
        )

    return tuple(
        run_stress_test(
            portfolio=portfolio,
            scenario=scenario,
        )
        for scenario in normalized_scenarios
    )


def main() -> None:
    """Run an illustrative portfolio-risk analysis."""

    portfolio = create_credit_portfolio(
        probabilities_of_default=[
            0.010,
            0.015,
            0.025,
            0.035,
            0.050,
            0.060,
            0.080,
            0.100,
        ],
        losses_given_default=[
            0.40,
            0.45,
            0.45,
            0.50,
            0.55,
            0.50,
            0.60,
            0.65,
        ],
        exposures_at_default=[
            2_000_000.0,
            1_500_000.0,
            1_200_000.0,
            900_000.0,
            750_000.0,
            600_000.0,
            400_000.0,
            300_000.0,
        ],
        identifiers=[
            "Corporate_A",
            "Corporate_B",
            "Retail_A",
            "Retail_B",
            "RealEstate_A",
            "RealEstate_B",
            "Technology_A",
            "Technology_B",
        ],
    )

    sectors = [
        "Corporate",
        "Corporate",
        "Retail",
        "Retail",
        "Real Estate",
        "Real Estate",
        "Technology",
        "Technology",
    ]

    analysis = analyze_portfolio_risk(
        portfolio=portfolio,
        sectors=sectors,
    )

    print("Credit portfolio risk analysis")
    print(f"Exposures: {analysis.exposure_count}")
    print(
        f"Total exposure: ${analysis.total_exposure:,.2f}"
    )
    print(
        f"Expected loss: ${analysis.expected_loss:,.2f}"
    )
    print(
        "Weighted-average PD: "
        f"{analysis.weighted_average_pd:.2%}"
    )
    print(
        "Weighted-average LGD: "
        f"{analysis.weighted_average_lgd:.2%}"
    )
    print(
        "Largest exposure share: "
        f"{analysis.largest_exposure_share:.2%}"
    )
    print(
        "HHI concentration: "
        f"{analysis.herfindahl_hirschman_index:.6f}"
    )
    print(
        "Normalized concentration: "
        f"{analysis.normalized_concentration_index:.2%}"
    )

    print("\nSector risk")

    for sector in analysis.sector_summaries:
        print(
            f"{sector.sector:>12}: "
            f"EAD=${sector.total_exposure:,.2f}, "
            f"EL=${sector.expected_loss:,.2f}, "
            f"EL share={sector.expected_loss_share:.2%}"
        )

    scenarios = [
        StressScenario(
            name="Moderate recession",
            pd_multiplier=1.50,
            lgd_multiplier=1.10,
            ead_multiplier=1.02,
        ),
        StressScenario(
            name="Severe recession",
            pd_multiplier=2.50,
            lgd_multiplier=1.25,
            ead_multiplier=1.10,
        ),
    ]

    print("\nStress testing")

    for result in run_stress_scenarios(
        portfolio=portfolio,
        scenarios=scenarios,
    ):
        print(f"\n{result.scenario.name}")
        print(
            "Stressed expected loss: "
            f"${result.stressed_expected_loss:,.2f}"
        )
        print(
            "Absolute increase: "
            f"${result.absolute_increase:,.2f}"
        )
        print(
            "Percentage increase: "
            f"{result.percentage_increase:.2%}"
        )


if __name__ == "__main__":
    main()