import numpy as np
import pandas as pd
import pytest

from bias_analysis.election_configs import get_election_config
from bias_analysis.modeling.train import (
    build_poststrat_frame,
    fit_glm,
    fit_glm_baseline,
    fit_ols,
    fit_ols_baseline,
    poststratify,
)

RNG = np.random.default_rng(42)
N = 80

BIAS_COLUMNS = [
    "bias_confirmation",
    "bias_anchoring",
    "bias_availability",
    "bias_social_desirability",
    "bias_acquiescence",
    "bias_demand_characteristics",
]


@pytest.fixture(scope="module")
def analysis_df() -> pd.DataFrame:
    """Synthetic analysis-ready DataFrame with all model columns."""
    return pd.DataFrame(
        {
            "positive_share": RNG.uniform(0.2, 0.8, N),
            "audience_mean_partisanship": RNG.uniform(-1.0, 1.0, N),
            "author_partisanship": RNG.uniform(-1.0, 1.0, N),
            "candidate_order": RNG.uniform(-0.5, 0.5, N),
            "formality_bias": RNG.uniform(-1.0, 1.0, N),
            "positive_ideology_score": RNG.uniform(0.0, 1.0, N),
            "negative_ideology_score": RNG.uniform(0.0, 1.0, N),
            "candidate_A_support_score": RNG.uniform(0.0, 1.0, N),
            "candidate_B_support_score": RNG.uniform(0.0, 1.0, N),
            "candidate_A_oppose_score": RNG.uniform(0.0, 1.0, N),
            "candidate_B_oppose_score": RNG.uniform(0.0, 1.0, N),
            "undirected_sentiment": RNG.uniform(-1.0, 1.0, N),
            "sentiment_intensity": RNG.uniform(0.0, 1.0, N),
            "toxicity_score": RNG.uniform(0.0, 0.3, N),
            **{col: RNG.integers(0, 2, N).astype(float) for col in BIAS_COLUMNS},
            "author_gender_male": RNG.uniform(0.0, 1.0, N),
            "author_age_30_39": RNG.uniform(0.0, 1.0, N),
            "author_age_40_over": RNG.uniform(0.0, 1.0, N),
            "log_duration": RNG.uniform(0.0, 7.0, N),
            "total_votes": RNG.integers(10, 1000, N),
        }
    )


# ---------------------------------------------------------------------------
# build_poststrat_frame
# ---------------------------------------------------------------------------


class TestBuildPoststratFrame:
    def test_returns_dataframe(self, analysis_df):
        ps = build_poststrat_frame(analysis_df, election="us20")
        assert isinstance(ps, pd.DataFrame)

    def test_correct_number_of_strata(self, analysis_df):
        cfg = get_election_config("us20")
        n_partisan = len(cfg["scwg"]["partisan_strata"])
        n_ideological = len(cfg["scwg"]["ideological_strata"])
        ps = build_poststrat_frame(analysis_df, election="us20")
        assert len(ps) == n_partisan * n_ideological

    def test_weights_sum_to_approximately_one(self, analysis_df):
        ps = build_poststrat_frame(analysis_df, election="us20")
        total_weight = ps["weight"].sum()
        assert abs(total_weight - 1.0) < 0.02

    def test_zero_bias_counterfactual_columns(self, analysis_df):
        ps = build_poststrat_frame(analysis_df, election="us20")
        for col in (
            "candidate_order",
            "formality_bias",
            "undirected_sentiment",
            "sentiment_intensity",
            "toxicity_score",
            *BIAS_COLUMNS,
        ):
            assert (ps[col] == 0.0).all(), f"Column '{col}' should be zero in post-strat frame"

    def test_required_predictor_columns_present(self, analysis_df):
        ps = build_poststrat_frame(analysis_df, election="us20")
        for col in (
            "audience_mean_partisanship",
            "author_partisanship",
            "positive_ideology_score",
            "author_gender_male",
            "author_age_30_39",
            "author_age_40_over",
            "weight",
        ):
            assert col in ps.columns, f"Missing column '{col}' in post-strat frame"

    @pytest.mark.parametrize("election", ["us16", "us20", "us24"])
    def test_works_for_all_elections(self, analysis_df, election):
        ps = build_poststrat_frame(analysis_df, election=election)
        assert len(ps) > 0
        assert ps["weight"].sum() > 0


# ---------------------------------------------------------------------------
# fit_glm / fit_glm_baseline
# ---------------------------------------------------------------------------


class TestFitGlm:
    def test_fit_glm_returns_result(self, analysis_df):
        result = fit_glm(analysis_df)
        assert result is not None

    def test_fit_glm_has_aic(self, analysis_df):
        result = fit_glm(analysis_df)
        assert isinstance(result.aic, float)
        assert np.isfinite(result.aic)

    def test_fit_glm_has_log_likelihood(self, analysis_df):
        result = fit_glm(analysis_df)
        assert np.isfinite(result.llf)

    def test_fit_glm_params_include_all_predictors(self, analysis_df):
        result = fit_glm(analysis_df)
        for col in ("candidate_order", "formality_bias", "positive_ideology_score"):
            assert col in result.params.index

    def test_fit_glm_baseline_has_fewer_params_than_full(self, analysis_df):
        full = fit_glm(analysis_df)
        baseline = fit_glm_baseline(analysis_df)
        assert len(full.params) > len(baseline.params)

    def test_fit_glm_baseline_excludes_bias_markers(self, analysis_df):
        baseline = fit_glm_baseline(analysis_df)
        for col in BIAS_COLUMNS:
            assert col not in baseline.params.index


# ---------------------------------------------------------------------------
# fit_ols / fit_ols_baseline
# ---------------------------------------------------------------------------


class TestFitOls:
    def test_fit_ols_returns_result(self, analysis_df):
        result = fit_ols(analysis_df)
        assert result is not None

    def test_fit_ols_has_rsquared_adj(self, analysis_df):
        result = fit_ols(analysis_df)
        assert isinstance(result.rsquared_adj, float)

    def test_fit_ols_has_aic(self, analysis_df):
        result = fit_ols(analysis_df)
        assert np.isfinite(result.aic)

    def test_fit_ols_baseline_has_fewer_params_than_full(self, analysis_df):
        full = fit_ols(analysis_df)
        baseline = fit_ols_baseline(analysis_df)
        assert len(full.params) > len(baseline.params)

    def test_fit_ols_params_include_bias_markers(self, analysis_df):
        result = fit_ols(analysis_df)
        for col in BIAS_COLUMNS:
            assert col in result.params.index


# ---------------------------------------------------------------------------
# poststratify
# ---------------------------------------------------------------------------


class TestPoststratify:
    def test_returns_float(self, analysis_df):
        glm_result = fit_glm(analysis_df)
        ps_frame = build_poststrat_frame(analysis_df, election="us20")
        estimate = poststratify(glm_result, ps_frame)
        assert isinstance(estimate, float)

    def test_estimate_is_valid_probability(self, analysis_df):
        glm_result = fit_glm(analysis_df)
        ps_frame = build_poststrat_frame(analysis_df, election="us20")
        estimate = poststratify(glm_result, ps_frame)
        assert 0.0 <= estimate <= 1.0

    def test_estimate_is_finite(self, analysis_df):
        glm_result = fit_glm(analysis_df)
        ps_frame = build_poststrat_frame(analysis_df, election="us20")
        estimate = poststratify(glm_result, ps_frame)
        assert np.isfinite(estimate)
