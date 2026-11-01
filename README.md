# Credit Risk Modeling Engine

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-237%20passed-brightgreen)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success)](#project-status)

A quantitative credit-risk analytics engine developed in Python for expected-loss estimation, credit scoring, rating migrations, correlated-default simulation, portfolio concentration analysis, stress testing, and tail-risk measurement.

> Educational quantitative-finance software. It is not a lending recommendation, accounting system, regulatory-capital model, or financial advisory service.

---

## English

### Overview

The Credit Risk Modeling Engine is a modular Python project for studying the main mathematical and computational components of credit risk.

It integrates individual borrower analysis with portfolio-level risk measurement using:

- Probability of Default (PD)
- Loss Given Default (LGD)
- Exposure at Default (EAD)
- Expected Loss (EL)
- Unexpected Loss (UL)
- Logistic-regression credit scoring
- Credit-rating transition matrices
- Multi-period default probabilities
- Gaussian one-factor correlated defaults
- Monte Carlo credit-loss simulation
- Credit Value at Risk
- Expected Shortfall
- Economic Capital
- Borrower and sector concentration
- Herfindahl–Hirschman Index
- Scenario and stress analysis
- Professional risk visualizations

The project emphasizes mathematical transparency, input validation, reproducibility, automated testing, and clear financial interpretation.

---

### Main objectives

The engine was designed to answer questions such as:

- What is the expected credit loss of an individual borrower?
- What is the total expected loss of a credit portfolio?
- How can borrower characteristics be converted into default probabilities?
- How may credit ratings migrate over time?
- What is the cumulative probability of default over several years?
- How do common economic shocks create correlated defaults?
- Where does the extreme-loss region begin?
- How severe are losses beyond Credit VaR?
- How much economic capital may be required?
- Is the portfolio concentrated in a few borrowers or sectors?
- How does the portfolio behave under recession scenarios?

---

### Core mathematical framework

#### Expected Loss

For borrower \(i\):

\[
EL_i = PD_i \times LGD_i \times EAD_i
\]

Portfolio expected loss is:

\[
EL_{\text{portfolio}}
=
\sum_{i=1}^{n}
PD_iLGD_iEAD_i
\]

#### Unexpected Loss

For a Bernoulli default exposure:

\[
UL_i =
LGD_iEAD_i
\sqrt{PD_i(1-PD_i)}
\]

#### Logistic default model

The probability of default is estimated as:

\[
PD_i
=
\frac{1}
{1+\exp(-z_i)}
\]

where:

\[
z_i =
\beta_0+
\beta_1x_{i1}
+\cdots+
\beta_kx_{ik}
\]

#### Rating transitions

A one-period transition matrix \(P\) contains:

\[
p_{ij}
=
P(R_{t+1}=j\mid R_t=i)
\]

Multi-period transition probabilities are obtained through:

\[
P^{(h)}=P^h
\]

#### Correlated defaults

Latent borrower credit quality is modeled as:

\[
A_i =
\sqrt{\rho_i}Z
+
\sqrt{1-\rho_i}\varepsilon_i
\]

Default occurs when:

\[
A_i < \Phi^{-1}(PD_i)
\]

#### Credit VaR

At confidence level \(\alpha\):

\[
VaR_{\alpha}(L)
=
\inf
\left\{
\ell:
P(L\leq\ell)\geq\alpha
\right\}
\]

#### Expected Shortfall

\[
ES_{\alpha}
=
E
\left[
L
\mid
L\geq VaR_{\alpha}(L)
\right]
\]

#### Economic Capital

\[
EC_{\alpha}
=
VaR_{\alpha}-EL
\]

#### Concentration

For exposure weights \(w_i\):

\[
HHI =
\sum_{i=1}^{n}w_i^2
\]

A complete derivation and interpretation of every model is available in [docs/methodology.md](docs/methodology.md).

---

### Project modules

| Module | Purpose |
|---|---|
| `expected_loss.py` | PD, LGD, EAD, expected loss, unexpected loss, and portfolio aggregation |
| `credit_scoring.py` | Logistic-regression credit scoring and classification metrics |
| `transition_matrix.py` | Rating migrations, multi-period probabilities, and simulated rating paths |
| `default_simulation.py` | Correlated-default Monte Carlo simulation and tail-risk measures |
| `portfolio_risk.py` | Exposure concentration, HHI, sector analysis, and stress testing |
| `visualization.py` | Professional credit-risk charts and diagnostic figures |

---

### Main capabilities

#### 1. Expected and unexpected loss

The engine calculates individual and portfolio credit losses using validated PD, LGD, and EAD inputs.

It distinguishes between:

- average credit cost;
- loss variability;
- borrower-level contribution;
- total portfolio exposure;
- expected-loss rates.

#### 2. Logistic credit scoring

The scoring component:

- trains a logistic-regression model;
- estimates borrower default probabilities;
- produces binary classifications;
- reports model coefficients;
- calculates accuracy, precision, recall, F1, and ROC-AUC;
- supports reproducible analytical workflows.

#### 3. Rating-transition analysis

The transition component:

- validates probability matrices;
- verifies that rows sum to one;
- supports an absorbing default state;
- calculates matrix powers;
- estimates multi-year migration probabilities;
- extracts cumulative default probabilities;
- simulates credit-rating paths.

#### 4. Correlated-default simulation

The Monte Carlo engine:

- preserves individual unconditional default probabilities;
- represents a common systematic economic factor;
- includes borrower-specific shocks;
- generates correlated default events;
- simulates portfolio losses;
- calculates Credit VaR;
- calculates Expected Shortfall;
- estimates economic capital;
- supports reproducible random seeds.

#### 5. Portfolio concentration

The portfolio component measures:

- borrower exposure shares;
- largest-borrower concentration;
- sector exposure;
- expected-loss contributions;
- Herfindahl–Hirschman Index;
- effective diversification.

#### 6. Stress testing

The stress-testing framework applies controlled shocks to:

- PD;
- LGD;
- EAD;
- economic sectors;
- correlation assumptions.

It compares baseline and stressed expected losses and helps identify the portfolio’s principal vulnerabilities.

#### 7. Visualization

The project generates professional charts for:

- expected-loss contributions;
- credit-rating transition matrices;
- simulated credit-loss distributions;
- Credit VaR and Expected Shortfall;
- borrower and sector concentration;
- stress scenarios;
- correlation sensitivity.

---

### Example analytical results

The included notebook demonstrates a portfolio with results such as:

| Risk indicator | Example result |
|---|---:|
| Total credit exposure | \$7,650,000.00 |
| Analytical expected loss | \$124,700.00 |
| Expected-loss rate | 1.63% |
| Credit VaR at 99% | \$1,147,500.00 |
| Expected Shortfall at 99% | \$1,431,976.92 |
| Economic Capital | \$1,022,800.00 |
| Weighted-average PD | 3.13% |
| Weighted-average LGD | 47.22% |
| Largest borrower share | 26.14% |
| HHI concentration | 0.165278 |
| Severe-recession expected loss | \$428,656.25 |

These values are generated from educational sample data and model assumptions. They are not forecasts for a real financial institution.

---

### Repository structure

```text
credit-risk-modeling-engine/
├── docs/
│   └── methodology.md
├── notebooks/
│   └── credit_risk_analysis.ipynb
├── results/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── credit_scoring.py
│   ├── default_simulation.py
│   ├── expected_loss.py
│   ├── portfolio_risk.py
│   ├── transition_matrix.py
│   └── visualization.py
├── tests/
│   ├── test_credit_scoring.py
│   ├── test_default_simulation.py
│   ├── test_expected_loss.py
│   ├── test_portfolio_risk.py
│   ├── test_transition_matrix.py
│   └── test_visualization.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/acarrillofintech/credit-risk-modeling-engine.git
cd credit-risk-modeling-engine
```

#### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### 4. Validate the environment

```bash
python -m pip check
```

---

### Running the modules

Expected-loss analysis:

```bash
python -m src.expected_loss
```

Credit-scoring model:

```bash
python -m src.credit_scoring
```

Rating-transition analysis:

```bash
python -m src.transition_matrix
```

Correlated-default simulation:

```bash
python -m src.default_simulation
```

Portfolio concentration and stress testing:

```bash
python -m src.portfolio_risk
```

Generate visualizations:

```bash
python -m src.visualization
```

---

### Running the notebook

Start Jupyter:

```bash
python -m jupyter notebook
```

Open:

```text
notebooks/credit_risk_analysis.ipynb
```

Select the project kernel:

```text
Python 3.14 (.venv Credit Risk)
```

Then execute:

```text
Run All
```

The notebook combines the mathematical theory, Python implementation, financial interpretation, tables, stress analysis, and professional charts.

---

### Testing

Run the complete automated test suite:

```bash
python -m pytest -q
```

Current result:

```text
237 passed
```

Run the tests with detailed output:

```bash
python -m pytest -v
```

Run an individual test module:

```bash
python -m pytest tests/test_transition_matrix.py -v
```

The test suite covers:

- known mathematical results;
- numerical consistency;
- invalid-input handling;
- boundary conditions;
- probability constraints;
- transition-matrix rules;
- seeded simulation reproducibility;
- tail-risk relationships;
- portfolio aggregation;
- visualization output.

---

### Technology stack

- Python 3.14
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter
- pytest
- Git and GitHub

---

### Model assumptions and limitations

The project uses simplified and transparent models. Important assumptions include:

- PD, LGD, and EAD are supplied or estimated consistently;
- the analysis horizon is common across exposures;
- LGD and EAD are deterministic unless stressed;
- default dependence follows a Gaussian one-factor structure;
- transition probabilities are time-homogeneous;
- rating evolution follows the Markov property;
- default is an absorbing state;
- portfolio composition remains fixed during the horizon.

Production implementations would require real institutional data, independent validation, calibration, backtesting, formal governance, regulatory review, security controls, and continuous monitoring.

See [docs/methodology.md](docs/methodology.md) for detailed assumptions, formulas, validation procedures, and limitations.

---

### Project status

The principal quantitative components are implemented:

- [x] Expected and unexpected loss
- [x] Logistic credit scoring
- [x] Credit-rating transitions
- [x] Multi-period default probabilities
- [x] Correlated-default simulation
- [x] Credit VaR and Expected Shortfall
- [x] Economic Capital
- [x] Concentration analysis
- [x] Stress testing
- [x] Visualizations
- [x] Automated tests
- [x] Analytical notebook
- [x] Mathematical methodology

Potential extensions:

- [ ] IFRS 9 lifetime expected credit loss
- [ ] Basel regulatory-capital calculations
- [ ] Macroeconomic PD models
- [ ] Stochastic LGD and EAD
- [ ] Multi-factor default dependence
- [ ] Model calibration and backtesting
- [ ] Interactive web dashboard
- [ ] Real portfolio database integration

---

## Español

### Descripción general

Credit Risk Modeling Engine es un proyecto modular desarrollado en Python para estudiar los principales componentes matemáticos y computacionales del riesgo de crédito.

El sistema integra el análisis individual de deudores con la medición del riesgo de una cartera mediante:

- Probabilidad de Incumplimiento (PD)
- Pérdida Dado el Incumplimiento (LGD)
- Exposición al Momento del Incumplimiento (EAD)
- Pérdida Esperada (EL)
- Pérdida Inesperada (UL)
- Calificación crediticia mediante regresión logística
- Matrices de transición de calificaciones
- Probabilidades de incumplimiento de varios periodos
- Simulación de incumplimientos correlacionados
- Simulación Monte Carlo
- Credit Value at Risk
- Expected Shortfall
- Capital económico
- Concentración por deudor y sector
- Índice Herfindahl–Hirschman
- Pruebas de estrés
- Visualizaciones profesionales

El proyecto prioriza la transparencia matemática, la validación de entradas, la reproducibilidad, las pruebas automatizadas y una interpretación financiera clara.

---

### Objetivos principales

El motor busca responder preguntas como:

- ¿Cuál es la pérdida esperada de un deudor?
- ¿Cuál es la pérdida esperada total de la cartera?
- ¿Cómo se convierten las características del cliente en una probabilidad de incumplimiento?
- ¿Cómo puede cambiar una calificación crediticia con el tiempo?
- ¿Cuál es la probabilidad acumulada de incumplimiento durante varios años?
- ¿Cómo producen los factores económicos incumplimientos simultáneos?
- ¿Dónde comienza la región de pérdidas extremas?
- ¿Qué tan graves son las pérdidas que superan el Credit VaR?
- ¿Cuánto capital económico podría necesitarse?
- ¿Está la cartera concentrada en pocos clientes o sectores?
- ¿Qué sucede durante una recesión?

---

### Componentes del proyecto

| Módulo | Función |
|---|---|
| `expected_loss.py` | Calcula PD, LGD, EAD, pérdida esperada e inesperada |
| `credit_scoring.py` | Construye el modelo de regresión logística |
| `transition_matrix.py` | Analiza migraciones crediticias y probabilidades multianuales |
| `default_simulation.py` | Simula incumplimientos correlacionados y pérdidas extremas |
| `portfolio_risk.py` | Mide concentración, contribuciones y escenarios de estrés |
| `visualization.py` | Genera gráficos profesionales de riesgo de crédito |

---

### Interpretación de las medidas

| Medida | Pregunta que responde |
|---|---|
| PD | ¿Qué probabilidad existe de que el cliente incumpla? |
| LGD | ¿Qué porcentaje se perdería si incumple? |
| EAD | ¿Cuánto dinero estará expuesto? |
| Pérdida esperada | ¿Cuál es el costo crediticio promedio? |
| Pérdida inesperada | ¿Cuánta incertidumbre existe alrededor del promedio? |
| Credit VaR | ¿Dónde comienza la región de pérdida extrema? |
| Expected Shortfall | ¿Qué tan grave es la pérdida dentro de esa región? |
| Capital económico | ¿Cuánto capital respalda la pérdida inesperada? |
| HHI | ¿Qué tan concentrada está la cartera? |
| Matriz de transición | ¿Cómo pueden cambiar las calificaciones? |
| Prueba de estrés | ¿Qué podría suceder bajo condiciones adversas? |

---

### Ejecución rápida

Activar el entorno virtual en Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

Ejecutar todas las pruebas:

```powershell
python -m pytest -q
```

Ejecutar la simulación de incumplimientos:

```powershell
python -m src.default_simulation
```

Ejecutar el análisis de cartera:

```powershell
python -m src.portfolio_risk
```

Abrir el notebook:

```powershell
python -m jupyter notebook
```

---

### Metodología matemática

La documentación matemática completa incluye:

- definiciones de PD, LGD y EAD;
- derivación de pérdida esperada;
- derivación de pérdida inesperada;
- regresión logística;
- métricas de clasificación;
- cadenas de Markov;
- matrices de transición;
- probabilidades acumuladas de incumplimiento;
- modelo gaussiano de un factor;
- algoritmo Monte Carlo;
- Credit VaR;
- Expected Shortfall;
- capital económico;
- concentración de cartera;
- índice HHI;
- pruebas de estrés;
- validación y limitaciones.

Puede consultarse en:

[docs/methodology.md](docs/methodology.md)

---

### Uso educativo

Este repositorio fue desarrollado como parte de un portafolio profesional de matemática financiera, ingeniería de software y análisis cuantitativo.

Su finalidad es demostrar:

- programación científica en Python;
- implementación de modelos financieros;
- diseño modular;
- pruebas automatizadas;
- simulación Monte Carlo;
- análisis estadístico;
- documentación matemática;
- interpretación de riesgo;
- control de versiones con Git.

---

### Aviso importante

Este software es un simulador cuantitativo con fines educativos.

Los resultados dependen de los datos, supuestos, parámetros y semillas aleatorias utilizados. No deben interpretarse como:

- recomendaciones de crédito;
- decisiones de aprobación o rechazo;
- provisiones contables oficiales;
- cálculos regulatorios;
- asesoría financiera;
- predicciones garantizadas.

El uso en producción requeriría datos institucionales validados, calibración, backtesting, gobierno de modelos, seguridad, auditoría y aprobación regulatoria.

---

## Author / Autor

**Alex Carrillo**

Systems Engineer and MSc in Financial Mathematics, focused on quantitative finance, financial risk, banking technology, data analytics, and Python software development.

GitHub: [acarrillofintech](https://github.com/acarrillofintech)

---

## License / Licencia

This project is distributed under the MIT License.

Este proyecto se distribuye bajo la licencia MIT.

See [LICENSE](LICENSE) for details.