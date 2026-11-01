"""Tests for credit-risk visualizations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

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
    create_transition_model,
)
from src.visualization import (
    CreditRiskFigurePaths,
    generate_credit_risk_figures,
    plot_exposure_contributions,
    plot_loss_distribution,
    plot_sector_risk,
    plot_stress_testing,
    plot_transition_heatmap,
)


@pytest.fixture(scope="module")
def portfolio() -> CreditPortfolio:
    """Create a portfolio for visualization tests."""

    return create_credit_portfolio(
        probabilities_of_default=[
            0.01,
            0.02,
            0.04,
            0.08,
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


@pytest.fixture(scope="module")
def simulation_result(
    portfolio: CreditPortfolio,
) -> DefaultSimulationResult:
    """Create reusable simulated losses."""

    return simulate_correlated_defaults(
        portfolio=portfolio,
        number_of_simulations=5_000,
        asset_correlation=0.20,
        confidence_level=0.99,
        random_state=42,
    )


@pytest.fixture(scope="module")
def portfolio_analysis(
    portfolio: CreditPortfolio,
) -> PortfolioRiskAnalysis:
    """Create reusable concentration analysis."""

    return analyze_portfolio_risk(
        portfolio=portfolio,
        sectors=[
            "Corporate",
            "Corporate",
            "Retail",
            "Retail",
        ],
    )


@pytest.fixture(scope="module")
def stress_results(
    portfolio: CreditPortfolio,
) -> tuple[StressTestResult, ...]:
    """Create reusable stress-test results."""

    return run_stress_scenarios(
        portfolio=portfolio,
        scenarios=[
            StressScenario(
                name="Moderate recession",
                pd_multiplier=1.50,
                lgd_multiplier=1.10,
            ),
            StressScenario(
                name="Severe recession",
                pd_multiplier=2.50,
                lgd_multiplier=1.25,
            ),
        ],
    )


@pytest.fixture(scope="module")
def transition_model() -> RatingTransitionModel:
    """Create a small transition model."""

    return create_transition_model(
        ratings=("A", "BBB", "BB", "B", "Default"),
        transition_matrix=np.array(
            [
                [0.85, 0.10, 0.04, 0.01, 0.00],
                [0.03, 0.80, 0.12, 0.04, 0.01],
                [0.00, 0.05, 0.75, 0.15, 0.05],
                [0.00, 0.01, 0.05, 0.74, 0.20],
                [0.00, 0.00, 0.00, 0.00, 1.00],
            ]
        ),
    )


def test_loss_distribution_returns_figure(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Loss-distribution plot should return a Figure."""

    figure = plot_loss_distribution(simulation_result)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert (
        figure.axes[0].get_title()
        == "Monte Carlo Credit Loss Distribution"
    )

    plt.close(figure)


def test_loss_distribution_contains_reference_lines(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Loss plot should contain EL, VaR, and ES lines."""

    figure = plot_loss_distribution(simulation_result)

    assert len(figure.axes[0].lines) == 3

    plt.close(figure)


def test_loss_distribution_custom_bins(
    simulation_result: DefaultSimulationResult,
) -> None:
    """Custom histogram bins should be accepted."""

    figure = plot_loss_distribution(
        simulation_result,
        bins=20,
    )

    assert isinstance(figure, Figure)

    plt.close(figure)


def test_transition_heatmap_returns_figure(
    transition_model: RatingTransitionModel,
) -> None:
    """Transition heatmap should return a Figure."""

    figure = plot_transition_heatmap(
        transition_model,
        periods=1,
    )

    assert isinstance(figure, Figure)
    assert "Transition Matrix" in figure.axes[0].get_title()
    assert figure.axes[0].get_xlabel() == "Future rating"
    assert figure.axes[0].get_ylabel() == "Current rating"

    plt.close(figure)


def test_multi_period_transition_heatmap(
    transition_model: RatingTransitionModel,
) -> None:
    """Multi-period heatmap should identify its horizon."""

    figure = plot_transition_heatmap(
        transition_model,
        periods=5,
    )

    assert "5 periods" in figure.axes[0].get_title()

    plt.close(figure)


def test_exposure_contributions_returns_figure(
    portfolio_analysis: PortfolioRiskAnalysis,
) -> None:
    """Borrower contribution plot should return a Figure."""

    figure = plot_exposure_contributions(
        portfolio_analysis
    )

    assert isinstance(figure, Figure)
    assert (
        figure.axes[0].get_title()
        == "Largest Expected-Loss Contributions"
    )

    plt.close(figure)


def test_exposure_contributions_respects_top_n(
    portfolio_analysis: PortfolioRiskAnalysis,
) -> None:
    """Contribution chart should limit displayed borrowers."""

    figure = plot_exposure_contributions(
        portfolio_analysis,
        top_n=2,
    )

    assert len(figure.axes[0].patches) == 2

    plt.close(figure)


def test_sector_risk_returns_figure(
    portfolio_analysis: PortfolioRiskAnalysis,
) -> None:
    """Sector-risk plot should return a Figure."""

    figure = plot_sector_risk(portfolio_analysis)

    assert isinstance(figure, Figure)
    assert (
        figure.axes[0].get_title()
        == "Credit Risk by Economic Sector"
    )

    plt.close(figure)


def test_sector_risk_contains_two_bars_per_sector(
    portfolio_analysis: PortfolioRiskAnalysis,
) -> None:
    """Each sector should show EAD and expected-loss shares."""

    figure = plot_sector_risk(portfolio_analysis)

    sector_count = len(
        portfolio_analysis.sector_summaries
    )

    assert len(figure.axes[0].patches) == sector_count * 2

    plt.close(figure)


def test_stress_testing_returns_figure(
    stress_results: tuple[StressTestResult, ...],
) -> None:
    """Stress-testing plot should return a Figure."""

    figure = plot_stress_testing(stress_results)

    assert isinstance(figure, Figure)
    assert (
        figure.axes[0].get_title()
        == "Expected Loss Under Stress Scenarios"
    )

    plt.close(figure)


def test_stress_testing_contains_base_and_scenarios(
    stress_results: tuple[StressTestResult, ...],
) -> None:
    """Stress chart should include base plus all scenarios."""

    figure = plot_stress_testing(stress_results)

    assert len(figure.axes[0].patches) == 3

    plt.close(figure)


def test_generate_all_figures(
    tmp_path: Path,
    simulation_result: DefaultSimulationResult,
    transition_model: RatingTransitionModel,
    portfolio_analysis: PortfolioRiskAnalysis,
    stress_results: tuple[StressTestResult, ...],
) -> None:
    """Complete generator should create five image files."""

    paths = generate_credit_risk_figures(
        simulation_result=simulation_result,
        transition_model=transition_model,
        portfolio_analysis=portfolio_analysis,
        stress_results=stress_results,
        output_directory=tmp_path,
    )

    assert isinstance(paths, CreditRiskFigurePaths)
    assert paths.loss_distribution.exists()
    assert paths.transition_matrix.exists()
    assert paths.exposure_contributions.exists()
    assert paths.sector_risk.exists()
    assert paths.stress_testing.exists()


def test_generated_files_are_not_empty(
    tmp_path: Path,
    simulation_result: DefaultSimulationResult,
    transition_model: RatingTransitionModel,
    portfolio_analysis: PortfolioRiskAnalysis,
    stress_results: tuple[StressTestResult, ...],
) -> None:
    """Generated PNG files should contain image data."""

    paths = generate_credit_risk_figures(
        simulation_result=simulation_result,
        transition_model=transition_model,
        portfolio_analysis=portfolio_analysis,
        stress_results=stress_results,
        output_directory=tmp_path,
    )

    for path in vars(paths).values():
        assert path.stat().st_size > 0


def test_generated_paths_are_absolute(
    tmp_path: Path,
    simulation_result: DefaultSimulationResult,
    transition_model: RatingTransitionModel,
    portfolio_analysis: PortfolioRiskAnalysis,
    stress_results: tuple[StressTestResult, ...],
) -> None:
    """Returned paths should be absolute."""

    paths = generate_credit_risk_figures(
        simulation_result=simulation_result,
        transition_model=transition_model,
        portfolio_analysis=portfolio_analysis,
        stress_results=stress_results,
        output_directory=tmp_path,
    )

    assert all(
        path.is_absolute()
        for path in vars(paths).values()
    )


@pytest.mark.parametrize(
    "invalid_bins",
    [0, 1, 4, -10],
)
def test_invalid_bins_raise_value_error(
    simulation_result: DefaultSimulationResult,
    invalid_bins: int,
) -> None:
    """Histogram requires at least five bins."""

    with pytest.raises(
        ValueError,
        match="bins must be at least 5",
    ):
        plot_loss_distribution(
            simulation_result,
            bins=invalid_bins,
        )


@pytest.mark.parametrize(
    "invalid_bins",
    [10.5, "20", True],
)
def test_non_integer_bins_raise_type_error(
    simulation_result: DefaultSimulationResult,
    invalid_bins: object,
) -> None:
    """Histogram bin count must be an integer."""

    with pytest.raises(
        TypeError,
        match="bins must be an integer",
    ):
        plot_loss_distribution(
            simulation_result,
            bins=invalid_bins,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_top_n",
    [0, -1, -10],
)
def test_invalid_top_n_raises_value_error(
    portfolio_analysis: PortfolioRiskAnalysis,
    invalid_top_n: int,
) -> None:
    """At least one borrower must be displayed."""

    with pytest.raises(
        ValueError,
        match="top_n must be at least 1",
    ):
        plot_exposure_contributions(
            portfolio_analysis,
            top_n=invalid_top_n,
        )


@pytest.mark.parametrize(
    "invalid_top_n",
    [2.5, "5", True],
)
def test_non_integer_top_n_raises_type_error(
    portfolio_analysis: PortfolioRiskAnalysis,
    invalid_top_n: object,
) -> None:
    """Top-N value must be an integer."""

    with pytest.raises(
        TypeError,
        match="top_n must be an integer",
    ):
        plot_exposure_contributions(
            portfolio_analysis,
            top_n=invalid_top_n,  # type: ignore[arg-type]
        )


def test_empty_stress_results_raise_value_error() -> None:
    """Stress chart requires at least one result."""

    with pytest.raises(
        ValueError,
        match="results cannot be empty",
    ):
        plot_stress_testing([])


def test_invalid_stress_result_raises_type_error() -> None:
    """Every stress item must be a StressTestResult."""

    with pytest.raises(
        TypeError,
        match="Every result must be a StressTestResult",
    ):
        plot_stress_testing(
            ["invalid"]  # type: ignore[list-item]
        )


def test_invalid_simulation_result_raises_type_error() -> None:
    """Loss chart requires simulation results."""

    with pytest.raises(
        TypeError,
        match="result must be a DefaultSimulationResult",
    ):
        plot_loss_distribution(
            "invalid"  # type: ignore[arg-type]
        )


def test_invalid_transition_model_raises_type_error() -> None:
    """Heatmap requires a transition model."""

    with pytest.raises(
        TypeError,
        match="model must be a RatingTransitionModel",
    ):
        plot_transition_heatmap(
            "invalid"  # type: ignore[arg-type]
        )


def test_invalid_analysis_raises_type_error() -> None:
    """Portfolio charts require a risk analysis."""

    with pytest.raises(
        TypeError,
        match="analysis must be a PortfolioRiskAnalysis",
    ):
        plot_sector_risk(
            "invalid"  # type: ignore[arg-type]
        )