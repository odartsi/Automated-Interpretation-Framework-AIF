# Automated Interpretation Framework (AIF)


An intelligent system for automatically interpreting X-ray diffraction (XRD) patterns in materials science. The framework evaluates multiple phase interpretations by combining statistical analysis, chemical composition matching, and large language model (LLM) evaluation to identify the most plausible interpretation of experimental XRD data.

<img width="2500" height="903" alt="AIF approach_with_chemical_horizontal_FINAL_newpalette" src="https://github.com/user-attachments/assets/625c78f2-6601-453a-b135-46c37a0ab813" />


## Overview

The Automated Interpretation Framework addresses a critical challenge in materials science: determining which phase interpretation is most likely correct when multiple valid interpretations exist for a single XRD pattern. The framework uses a Bayesian approach, combining multiple evidence sources to calculate posterior probabilities for each interpretation.

### Key Features

- **Multi-factor Evaluation**: Combines statistical fit quality, chemical composition balance, and thermodynamic plausibility
- **LLM-powered Analysis**: Uses GPT-4 to evaluate phase stability and synthesis plausibility based on experimental conditions
- **Iterative Phase Search**: Systematically explores phase combinations by excluding candidate phases to discover alternative interpretations
- **Comprehensive Scoring**: Calculates multiple metrics including RWP, peak matching scores, background analysis, and composition balance
- **Visualization**: Generates detailed plots showing interpretation probabilities, phase contributions, and fit quality

## How It Works

The framework evaluates interpretations using a Bayesian approach:

```
P(I_n | S) ∝ P(S | I_n) × P(I_n)
```

Where:
- **P(I_n | S)**: Posterior probability of interpretation I_n given spectrum S
- **P(S | I_n)**: Statistical likelihood (fit quality) - how well the interpretation fits the XRD pattern
- **P(I_n)**: Prior probability based on chemical and thermodynamic plausibility

### Evaluation Components

1. **Refinement metrics** (`refinement_metrics.py`):
   - Performs XRD phase search and Rietveld refinement
   - Computes RWP (weighted profile R-factor) and peak-matching score
   - Matches calculated vs observed peaks
   - Analyzes background and signal quality
   - Generates multiple interpretations by iteratively excluding phases

2. **Composition balance** (`composition_balance.py`):
   - Compares target composition with interpretation composition
   - Calculates composition balance scores (how well phases match target stoichiometry)
   - Normalizes compositions for fair comparison

3. **LLM Evaluation** (`LLM_evaluation.py`):
   - Uses GPT-4 to evaluate phase stability under synthesis conditions
   - Considers temperature, atmosphere, precursors, and dwell time
   - Evaluates polymorph stability and thermodynamic plausibility
   - Provides explanations for likelihood scores

4. **Probability Calculation** (`probability_of_interpretation.py`):
   - Combines all factors into prior and posterior probabilities
   - Normalizes scores across interpretations
   - Flags untrustworthy interpretations
   - Generates visualization plots

## Project Structure

```
AIF_copy/
├── src/
│   ├── refinement_metrics.py      # RWP, score, refinement and fit metrics
│   ├── composition_balance.py     # Composition balance scores
│   ├── LLM_evaluation.py            # LLM-based phase evaluation
│   ├── probability_of_interpretation.py  # Main orchestration script
│   ├── utils.py                    # Utility functions and plotting
│   ├── prompts/
│   │   └── llm_prompt_template.txt # LLM prompt template
│   └── tests/                      # Test scripts and comparisons
├── data/
│   ├── alab_synthesis_data/        # Synthesis metadata (CSV files)
│   └── xrd_data/                   # XRD patterns and interpretations
├── notebooks/                      # Jupyter notebooks for analysis
└── README.md
```

## Dependencies

### Core Libraries
- **pymatgen**: Materials analysis and composition handling
- **openai**: LLM API access (via CBORG gateway)
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib/plotly**: Visualization

### The `dara` package (required for refinement and phase search)

The framework depends on **dara**, a library used for:

- **XRD phase search** (`dara.search_phases`, `dara.structure_db.ICSDDatabase`)
- **Rietveld refinement** (`dara.refine.do_refinement_no_saving`)
- **Peak matching and detection** (`dara.search.peak_matcher.PeakMatcher`, `dara.peak_detection.detect_peaks`)
- **Refinement results** (`dara.result.RefinementResult` in `utils.py`)

Without `dara` installed and on your `PYTHONPATH`, the main entry point (`probability_of_interpretation.py`) and refinement metrics (`refinement_metrics.py`) will not run. Ensure you have access to the dara package and that it is installed in the same environment you use to run this code. (If dara is internal or versioned elsewhere, install it according to your organization’s instructions.)

### External Services
- **CBORG Gateway**: API access to GPT-4 (`https://api.cborg.lbl.gov`)
- **ICSD Database**: Crystal structure database access (via dara)

## Installation and environment

1. **Python environment**: Use a Python environment where you can install the dependencies (Python 3.8+ recommended).

2. **Install dependencies**: Install pymatgen, openai, pandas, numpy, matplotlib, plotly (e.g. with `pip` or your project’s environment manager).

3. **Install and expose `dara`**: The `dara` package must be available when running the code. Install it from your usual source (internal repo, PyPI, etc.) and ensure it is on `PYTHONPATH` or installed into the same environment.

4. **Run from the project root**: Execute the main script from the repository root so that imports like `from utils import ...` and data paths resolve correctly:
   ```bash
   cd /path/to/AIF_copy
   python src/probability_of_interpretation.py
   ```

5. **Optional – run unit tests**: Install pytest and run tests from the project root (with `src` on `PYTHONPATH`):
   ```bash
   pip install -r requirements-dev.txt
   PYTHONPATH=src python -m pytest tests/ -v
   ```
   Unit tests live in the top-level `tests/` directory and cover pure functions in `utils`, `composition_balance`, `refinement_metrics`, `LLM_evaluation`, and `probability_of_interpretation` (e.g. `load_json`, `extract_project_number_from_filename`, `scaled_sigmoid`, `calculate_posterior_probability_of_interpretation`, `normalize_rwp`, `_infer_group_from_search`).

6. **API key for LLM**: Set your API key for the CBORG gateway (see [Environment Setup](#environment-setup) below).

## Usage

Before running, ensure you have completed [Installation and environment](#installation-and-environment) (including the `dara` package and API key).

### Basic Workflow

1. **Prepare Data**:
   - XRD pattern files (`.xy` or `.xrdml` format)
   - Synthesis metadata CSV with columns: `Name`, `Target`, `Precursors`, `Temperature (C)`, `Dwell Duration (h)`, `Furnace`
   - JSON file mapping patterns to chemical systems

2. **Run Evaluation**:
   ```python
   python src/probability_of_interpretation.py
   ```
   
   The script will prompt you to:
   - Choose dataset group (TRI, MINES, ARR, GENOME, etc.)
   - Select all patterns or specific ones
   - Process each pattern and generate interpretations

3. **View Results**:
   - Interpretation probabilities are saved to JSON files
   - Plots are generated showing:
     - Interpretation probabilities
     - Phase probabilities
     - Contribution decomposition
     - Fit quality metrics

### Dataset Configuration

The framework supports multiple datasets configured in `probability_of_interpretation.py`:

```python
DATASETS = {
    "TRI": {
        "csv": "../data/alab_synthesis_data/synthesis_TRI.csv",
        "combos": "../data/xrd_data/combinations.json",
        "interpretations": "../data/xrd_data/interpretations/interpretations_for_brier.json"
    },
    # ... other datasets
}
```

### Environment Setup

Set the API key for LLM access:
```bash
export API_KEY="your-api-key-here"
```

## Key Metrics

### Statistical Metrics
- **RWP**: Weighted profile R-factor (lower is better)
- **Peak Score**: Matches calculated vs observed peaks
- **Missing/Extra Peaks**: Count of unmatched peaks
- **Background Quality**: Analysis of background fitting

### Chemical Metrics
- **Composition Balance Score**: How well interpretation matches target (0-1)
- **Gas Loss Adjustment**: Accounts for volatile species

### LLM Metrics
- **Phase Likelihood**: Individual phase plausibility (0-1)
- **Interpretation Likelihood**: Overall interpretation plausibility (0-1)
- **Explanations**: Detailed reasoning for scores

### Final Output
- **Prior Probability**: Combined chemical and LLM scores
- **Posterior Probability**: Final probability after statistical weighting
- **Trust Score**: Flag for potentially unreliable interpretations

## Example Output

For each interpretation, the framework provides:

```
Interpretation: I_1
  Phases: ZrTiO4_60, ZrO2_14
  LLM Interpretation Likelihood: 93.0%
  Composition Balance: 90.0%
  Fit quality: 0.85
  Normalized_rwp: 75.0%
  Normalized_score: 82.0%
  Prior Probability: 88.5%
  Posterior Probability: 92.3%
```

## Advanced Features

### Iterative Phase Exploration

The framework systematically explores alternative interpretations by:
1. Starting with initial phase search results
2. Removing individual phases and re-refining
3. Testing combinations of excluded phases
4. Tracking unique phase combinations to avoid duplicates

### Customizable Weights

Adjust the importance of different factors:
```python
interpretations = calculate_prior_probability(
    interpretations, 
    w_llm=0.5,      # LLM weight
    w_bscore=0.7    # Balance score weight
)

interpretations = calculate_fit_quality(
    interpretations,
    w_rwp=1,        # RWP weight
    w_score=0.5     # Peak score weight
)
```

## Testing

Test scripts are available in `src/tests/`:
- `compare_AIF_Dara_selections.py`: Compare AIF selections with expert selections
- `check_if_AIF_selected_the_best.py`: Verify best interpretation selection
- `rank_interpretations.py`: Rank interpretations by various metrics

## Limitations

- Requires the **dara** package for XRD refinement and phase search; the code will not run without it.
- Requires API access to LLM service (CBORG gateway)
- Processing time scales with number of interpretations (typically 10-20 minutes per pattern)
- Quality depends on completeness of synthesis metadata
- LLM evaluation may vary slightly between runs (uses seed=42 for reproducibility)

## Citation

If you use this framework in your research, please cite appropriately.

## License

[Specify license if applicable]

## Contact

[Add contact information]
