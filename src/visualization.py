"""Professional visualizations for the credit-risk engine."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from src.default_simulation import (
    CreditPortfolio,
    DefaultSimulationResult,
    create_credit_portfolio,
    simulate_correlated_defaults,
)
from src.portfolio_risk import (
    PortfolioRiskAnalysis,
    StressScenario,
    StressTestResult,
    analyze_portfolio_risk,
    run_stress_scenarios,
)
from src.transition_matrix import (
    RatingTransitionModel,
    calculate_transition_matrix,
    create_transition_model,
)


@dataclass(frozen=True)
class CreditRiskFigurePaths:
    """Locations of generated credit-risk figures."""

    loss_distribution: Path
    transition_matrix: Path
    exposure_contributions: Path
    sector_risk: Path
    stress_testing: Path


def _currency_formatter(
    value: float,
    _position: int,
) -> str:
    """Format monetary chart axes."""

    absolute_value = abs(value)

    if absolute_value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"

    if absolute_value >= 1_000:
        return f"${value / 1_000:.0f}K"

    return f"${value:,.0f}"


def _apply_style() -> None:
    """Apply the common visualization style."""

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="deep",
    )


def plot_loss_distribution(
    result: DefaultSimulationResult,
    bins: int = 50,
) -> Figure:
    """Plot simulated portfolio losses with EL, VaR, and ES."""

    if not isinstance(result, DefaultSimulationResult):
        raise TypeError(
            "result must be a DefaultSimulationResult."
        )

    if isinstance(bins, bool) or not isinstance(bins, Integral):
        raise TypeError("bins must be an integer.")

    normalized_bins = int(bins)

    if normalized_bins < 5:
        raise ValueError("bins must be at least 5.")

    _apply_style()

    figure, axis = plt.subplots(figsize=(12, 7))

    sns.histplot(
        result.loss_distribution,
        bins=normalized_bins,
        stat="density",
        color="#4C72B0",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
        ax=axis,
    )

    axis.axvline(
        result.analytical_expected_loss,
        color="#55A868",
        linewidth=2.2,
        linestyle="--",
        label=(
            "Expected Loss "
            f"(${result.analytical_expected_loss:,.0f})"
        ),
    )
    axis.axvline(
        result.value_at_risk,
        color="#C44E52",
        linewidth=2.2,
        linestyle="-",
        label=(
            f"Credit VaR {result.confidence_level:.0%} "
            f"(${result.value_at_risk:,.0f})"
        ),
    )
    axis.axvline(
        result.expected_shortfall,
        color="#8172B2",
        linewidth=2.2,
        linestyle=":",
        label=(
            "Expected Shortfall "
            f"(${result.expected_shortfall:,.0f})"
        ),
    )

    axis.set_title(
        "Monte Carlo Credit Loss Distribution",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    axis.set_xlabel("Portfolio loss")
    axis.set_ylabel("Probability density")
    axis.xaxis.set_major_formatter(
        FuncFormatter(_currency_formatter)
    )
    axis.legend(frameon=True)
    axis.grid(alpha=0.25)

    figure.tight_layout()

    return figure


def plot_transition_heatmap(
    model: RatingTransitionModel,
    periods: int = 1,
) -> Figure:
    """Plot a rating-transition probability heatmap."""

    if not isinstance(model, RatingTransitionModel):
        raise TypeError(
            "model must be a RatingTransitionModel."
        )

    matrix = calculate_transition_matrix(
        model=model,
        periods=periods,
    )

    _apply_style()

    figure, axis = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        xticklabels=model.ratings,
        yticklabels=model.ratings,
        cbar_kws={
            "label": "Transition probability",
        },
        ax=axis,
    )

    period_label = (
        "period"
        if periods == 1
        else "periods"
    )

    axis.set_title(
        f"Credit Rating Transition Matrix — "
        f"{periods} {period_label}",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    axis.set_xlabel("Future rating")
    axis.set_ylabel("Current rating")
    axis.tick_params(
        axis="x",
        rotation=45,
    )
    axis.tick_params(
        axis="y",
        rotation=0,
    )

    figure.tight_layout()

    return figure


def plot_exposure_contributions(
    analysis: PortfolioRiskAnalysis,
    top_n: int = 10,
) -> Figure:
    """Plot the largest expected-loss contributions."""

    if not isinstance(analysis, PortfolioRiskAnalysis):
        raise TypeError(
            "analysis must be a PortfolioRiskAnalysis."
        )

    if isinstance(top_n, bool) or not isinstance(top_n, Integral):
        raise TypeError("top_n must be an integer.")

    normalized_top_n = int(top_n)

    if normalized_top_n < 1:
        raise ValueError("top_n must be at least 1.")

    contributions = sorted(
        analysis.exposure_contributions,
        key=lambda item: item.expected_loss,
        reverse=True,
    )[:normalized_top_n]

    labels = [
        contribution.identifier
        for contribution in contributions
    ]
    values = [
        contribution.expected_loss
        for contribution in contributions
    ]

    _apply_style()

    figure, axis = plt.subplots(figsize=(12, 7))

    colors = sns.color_palette(
        "crest",
        n_colors=len(contributions),
    )

    bars = axis.barh(
        labels[::-1],
        values[::-1],
        color=colors[::-1],
        edgecolor="white",
    )

    axis.bar_label(
        bars,
        labels=[
            f"${value:,.0f}"
            for value in values[::-1]
        ],
        padding=5,
        fontsize=9,
    )

    axis.set_title(
        "Largest Expected-Loss Contributions",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    axis.set_xlabel("Expected loss")
    axis.set_ylabel("Credit exposure")
    axis.xaxis.set_major_formatter(
        FuncFormatter(_currency_formatter)
    )
    axis.grid(
        axis="x",
        alpha=0.25,
    )
    axis.grid(
        axis="y",
        visible=False,
    )

    figure.tight_layout()

    return figure


def plot_sector_risk(
    analysis: PortfolioRiskAnalysis,
) -> Figure:
    """Compare exposure and expected-loss shares by sector."""

    if not isinstance(analysis, PortfolioRiskAnalysis):
        raise TypeError(
            "analysis must be a PortfolioRiskAnalysis."
        )

    sectors = sorted(
        analysis.sector_summaries,
        key=lambda item: item.expected_loss_share,
        reverse=True,
    )

    labels = [
        sector.sector
        for sector in sectors
    ]
    exposure_shares = np.array(
        [
            sector.exposure_share
            for sector in sectors
        ]
    )
    expected_loss_shares = np.array(
        [
            sector.expected_loss_share
            for sector in sectors
        ]
    )

    positions = np.arange(len(labels))
    bar_width = 0.36

    _apply_style()

    figure, axis = plt.subplots(figsize=(12, 7))

    axis.bar(
        positions - bar_width / 2,
        exposure_shares,
        width=bar_width,
        color="#4C72B0",
        label="Exposure share",
    )
    axis.bar(
        positions + bar_width / 2,
        expected_loss_shares,
        width=bar_width,
        color="#C44E52",
        label="Expected-loss share",
    )

    axis.set_title(
        "Credit Risk by Economic Sector",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    axis.set_xlabel("Economic sector")
    axis.set_ylabel("Portfolio share")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: f"{value:.0%}"
        )
    )
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.grid(
        axis="x",
        visible=False,
    )

    figure.tight_layout()

    return figure


def plot_stress_testing(
    results: Sequence[StressTestResult],
) -> Figure:
    """Plot expected loss across stress scenarios."""

    if isinstance(results, (str, bytes)):
        raise TypeError(
            "results must be a sequence of StressTestResult objects."
        )

    normalized_results = tuple(results)

    if not normalized_results:
        raise ValueError("results cannot be empty.")

    if any(
        not isinstance(result, StressTestResult)
        for result in normalized_results
    ):
        raise TypeError(
            "Every result must be a StressTestResult."
        )

    base_expected_loss = (
        normalized_results[0].base_expected_loss
    )

    labels = ["Base"] + [
        result.scenario.name
        for result in normalized_results
    ]
    values = [base_expected_loss] + [
        result.stressed_expected_loss
        for result in normalized_results
    ]

    _apply_style()

    figure, axis = plt.subplots(figsize=(12, 7))

    colors = ["#4C72B0"] + [
        "#DD8452",
        "#C44E52",
        "#8172B2",
        "#937860",
    ][:len(normalized_results)]

    if len(colors) < len(values):
        colors.extend(
            sns.color_palette(
                "rocket",
                n_colors=len(values) - len(colors),
            )
        )

    bars = axis.bar(
        labels,
        values,
        color=colors,
        edgecolor="white",
        width=0.65,
    )

    axis.bar_label(
        bars,
        labels=[
            f"${value:,.0f}"
            for value in values
        ],
        padding=5,
        fontsize=9,
    )

    axis.set_title(
        "Expected Loss Under Stress Scenarios",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    axis.set_xlabel("Economic scenario")
    axis.set_ylabel("Expected loss")
    axis.yaxis.set_major_formatter(
        FuncFormatter(_currency_formatter)
    )
    axis.tick_params(
        axis="x",
        rotation=20,
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.grid(
        axis="x",
        visible=False,
    )

    figure.tight_layout()

    return figure


def _save_figure(
    figure: Figure,
    output_path: Path,
) -> Path:
    """Save one figure and return its absolute path."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    return output_path.resolve()


def generate_credit_risk_figures(
    simulation_result: DefaultSimulationResult,
    transition_model: RatingTransitionModel,
    portfolio_analysis: PortfolioRiskAnalysis,
    stress_results: Sequence[StressTestResult],
    output_directory: str | Path = "results/figures",
) -> CreditRiskFigurePaths:
    """Generate and save the complete figure collection."""

    output_path = Path(output_directory)

    loss_figure = plot_loss_distribution(
        simulation_result
    )
    transition_figure = plot_transition_heatmap(
        transition_model,
        periods=1,
    )
    contribution_figure = plot_exposure_contributions(
        portfolio_analysis
    )
    sector_figure = plot_sector_risk(
        portfolio_analysis
    )
    stress_figure = plot_stress_testing(
        stress_results
    )

    paths = CreditRiskFigurePaths(
        loss_distribution=_save_figure(
            loss_figure,
            output_path / "credit_loss_distribution.png",
        ),
        transition_matrix=_save_figure(
            transition_figure,
            output_path / "rating_transition_matrix.png",
        ),
        exposure_contributions=_save_figure(
            contribution_figure,
            output_path / "exposure_risk_contributions.png",
        ),
        sector_risk=_save_figure(
            sector_figure,
            output_path / "sector_credit_risk.png",
        ),
        stress_testing=_save_figure(
            stress_figure,
            output_path / "credit_stress_testing.png",
        ),
    )

    plt.close(loss_figure)
    plt.close(transition_figure)
    plt.close(contribution_figure)
    plt.close(sector_figure)
    plt.close(stress_figure)

    return paths


def main() -> None:
    """Generate the complete credit-risk figure collection."""

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

    portfolio_analysis = analyze_portfolio_risk(
        portfolio=portfolio,
        sectors=sectors,
    )

    stress_results = run_stress_scenarios(
        portfolio=portfolio,
        scenarios=[
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
        ],
    )

    simulation_result = simulate_correlated_defaults(
        portfolio=portfolio,
        number_of_simulations=50_000,
        asset_correlation=0.20,
        confidence_level=0.99,
        random_state=42,
    )

    ratings = (
        "AAA",
        "AA",
        "A",
        "BBB",
        "BB",
        "B",
        "Default",
    )

    transition_model = create_transition_model(
        ratings=ratings,
        transition_matrix=np.array(
            [
                [0.900, 0.080, 0.015, 0.004, 0.001, 0.000, 0.000],
                [0.020, 0.880, 0.075, 0.020, 0.004, 0.001, 0.000],
                [0.005, 0.040, 0.860, 0.075, 0.015, 0.004, 0.001],
                [0.000, 0.005, 0.050, 0.820, 0.090, 0.030, 0.005],
                [0.000, 0.000, 0.005, 0.060, 0.780, 0.130, 0.025],
                [0.000, 0.000, 0.000, 0.010, 0.080, 0.810, 0.100],
                [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
            ]
        ),
    )

    paths = generate_credit_risk_figures(
        simulation_result=simulation_result,
        transition_model=transition_model,
        portfolio_analysis=portfolio_analysis,
        stress_results=stress_results,
    )

    print("Credit-risk figures generated successfully")

    for field_name, path in vars(paths).items():
        print(f"{field_name}: {path}")


if __name__ == "__main__":
    main()