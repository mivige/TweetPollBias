"""
Sample-Calibrated Weighted GLM (SCWG) for Twitter Poll Bias Adjustment.

This module implements a frequentist SCGW pipeline that:
1. Merges raw poll data with project bias markers (candidate order, formality, political leaning).
2. Fits a Binomial GLM weighted by poll size.
3. Post-stratifies predictions over population strata.
4. Benchmarks the adjusted estimate against the actual election result.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from loguru import logger
from tqdm import tqdm
import typer

from bias_analysis.config import get_election_paths
from bias_analysis.election_configs import get_election_config
from bias_analysis.dataset import get_base_dataset

app = typer.Typer()

DEFAULT_ELECTION = "us20"


def load_and_merge_features(base_df: pd.DataFrame, election: str = DEFAULT_ELECTION) -> pd.DataFrame:
    """
    Merge the three processed feature CSVs into the base poll dataframe.
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    cand_pos = ecfg["bias_direction"]["positive"]
    cand_neg = ecfg["bias_direction"]["negative"]
    pos_lower = cand_pos.lower()
    neg_lower = cand_neg.lower()

    df = base_df.copy()
    df["tweet_id"] = df["tweet_id"].astype(str)

    co_path = paths.processed_dir / "candidate_order_features.csv"
    if co_path.exists():
        co_df = pd.read_csv(co_path)
        co_df["poll_id"] = co_df["poll_id"].astype(str)
        df = df.merge(
            co_df[["poll_id", f"{cand_pos}_position", f"{cand_neg}_position",
                   f"{cand_pos}_percentage", f"{cand_neg}_percentage", 
                   f"{cand_pos}_votes", f"{cand_neg}_votes"]],
            left_on="tweet_id", right_on="poll_id", how="left",
        )
        df["candidate_order"] = (
            (df[f"{cand_neg}_position"] - df[f"{cand_pos}_position"]) * 0.25
        )
        missing_mask = df[f"{cand_neg}_position"].isna() | df[f"{cand_pos}_position"].isna()
        df.loc[missing_mask, "candidate_order"] = np.nan
    else:
        logger.warning(f"Missing {co_path.name} — candidate_order will be NaN")
        for col in [f"{cand_pos}_percentage", f"{cand_neg}_percentage",
                     f"{cand_pos}_votes", f"{cand_neg}_votes"]:
            df[col] = np.nan
        df["candidate_order"] = np.nan

    fi_path = paths.processed_dir / "formal_informal_appellatives.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        fi_df["poll_id"] = fi_df["poll_id"].astype(str)

        # Prefer new formality_score columns (0-6 van den Berg scale)
        pos_score_col = f"{cand_pos}_formality_score"
        neg_score_col = f"{cand_neg}_formality_score"
        if pos_score_col in fi_df.columns and neg_score_col in fi_df.columns:
            df = df.merge(
                fi_df[["poll_id", pos_score_col, neg_score_col]],
                left_on="tweet_id", right_on="poll_id", how="left",
                suffixes=("", "_fi"),
            )
            # formality_bias in [-1, +1]: (pos_score - neg_score) / 6
            pos_f = df[pos_score_col].fillna(3.0)  # 3 = LN (neutral midpoint)
            neg_f = df[neg_score_col].fillna(3.0)
            df["formality_bias"] = np.clip((pos_f - neg_f) / 6.0, -1.0, 1.0)
        else:
            # Legacy fallback: old {C}_label columns with formal/informal/neutral
            merge_cols = ["poll_id"]
            if f"{cand_pos}_label" in fi_df.columns:
                merge_cols.append(f"{cand_pos}_label")
            if f"{cand_neg}_label" in fi_df.columns:
                merge_cols.append(f"{cand_neg}_label")
            df = df.merge(
                fi_df[merge_cols],
                left_on="tweet_id", right_on="poll_id", how="left",
                suffixes=("", "_fi"),
            )
            label_map = {"formal": 1.0, "informal": -1.0, "neutral": 0.0}
            pos_formal = df.get(f"{cand_pos}_label", pd.Series(dtype=float)).map(label_map).fillna(0.0)
            neg_formal = df.get(f"{cand_neg}_label", pd.Series(dtype=float)).map(label_map).fillna(0.0)
            df["formality_bias"] = np.clip(pos_formal - neg_formal, -1.0, 1.0)
    else:
        logger.warning(f"Missing {fi_path.name} — formality_bias will be 0")
        df["formality_bias"] = 0.0

    pl_path = paths.processed_dir / "political_leaning_features.csv"
    pos_score_col = ecfg["hypothesis_column_mapping"].get("positive_ideology", "positive_ideology_score")
    if pl_path.exists():
        pl_df = pd.read_csv(pl_path)
        pl_df["poll_id"] = pl_df["poll_id"].astype(str)
        df = df.merge(
            pl_df[["poll_id", pos_score_col]],
            left_on="tweet_id", right_on="poll_id", how="left",
            suffixes=("", "_pl"),
        )
        df = df.rename(columns={pos_score_col: "positive_ideology_score"})
    else:
        logger.warning(f"Missing {pl_path.name} — positive_ideology_score will be 0")
        df["positive_ideology_score"] = 0.0

    st_path = paths.processed_dir / "sentiment_toxicity_features.csv"
    if st_path.exists():
        st_df = pd.read_csv(st_path)
        st_df["poll_id"] = st_df["poll_id"].astype(str)
        df = df.merge(
            st_df[["poll_id", "undirected_sentiment", "sentiment_intensity", "toxicity_score"]],
            left_on="tweet_id", right_on="poll_id", how="left",
            suffixes=("", "_st"),
        )
    else:
        logger.warning(f"Missing {st_path.name} — sentiment and toxicity will be 0")
        df["undirected_sentiment"] = 0.0
        df["sentiment_intensity"] = 0.0
        df["toxicity_score"] = 0.0

    cb_path = paths.processed_dir / "cognitive_biases.jsonl"
    bias_columns = [
        "bias_confirmation", "bias_anchoring", "bias_availability",
        "bias_social_desirability", "bias_acquiescence", "bias_demand_characteristics"
    ]
    if cb_path.exists():
        bias_data = []
        with open(cb_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    poll_id = str(record.get("id", ""))
                    biases = record.get("biases_detected", [])
                    
                    bias_dict = {"poll_id": poll_id}
                    for b in bias_columns:
                        bias_dict[b] = 0.0
                        
                    for b_info in biases:
                        b_type = str(b_info.get("bias_type", "")).lower()
                        if "confirmation" in b_type: bias_dict["bias_confirmation"] = 1.0
                        elif "anchor" in b_type: bias_dict["bias_anchoring"] = 1.0
                        elif "availab" in b_type: bias_dict["bias_availability"] = 1.0
                        elif "social" in b_type: bias_dict["bias_social_desirability"] = 1.0
                        elif "acquiescence" in b_type: bias_dict["bias_acquiescence"] = 1.0
                        elif "demand" in b_type: bias_dict["bias_demand_characteristics"] = 1.0
                        
                    bias_data.append(bias_dict)
                except Exception as e:
                    pass
        
        cb_df = pd.DataFrame(bias_data)
        if not cb_df.empty:
            df = df.merge(cb_df, left_on="tweet_id", right_on="poll_id", how="left", suffixes=("", "_cb"))
        else:
            for b in bias_columns:
                df[b] = 0.0
    else:
        logger.warning(f"Missing {cb_path.name} — cognitive biases will be 0")
        for b in bias_columns:
            df[b] = 0.0

    # --- Target: positive_share (Head-to-Head) ------------------------------
    pos_votes = df[f"{cand_pos}_votes"].fillna(0)
    neg_votes = df[f"{cand_neg}_votes"].fillna(0)
    h2h_total = pos_votes + neg_votes
    
    # Calculate share and override total_votes to reflect H2H sample size
    df["positive_share"] = np.where(h2h_total > 0, pos_votes / h2h_total, np.nan)
    df["total_votes"] = h2h_total

    for col in ["audience_mean_partisanship", "author_partisanship"]:
        if col in df.columns:
            col_mean = df[col].mean()
            fill_value = col_mean if pd.notna(col_mean) else 0.0
            df[col] = df[col].fillna(fill_value)
        else:
            df[col] = 0.0

    for col in ["candidate_order", "formality_bias", "positive_ideology_score"]:
        df[col] = df[col].fillna(0.0)

    for col in ["author_gender_male", "author_age_30_39", "author_age_40_over"]:
        if col in df.columns:
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean if pd.notna(col_mean) else 0.5)
        else:
            df[col] = 0.5

    # Rows with missing positive_share will be dropped later.
    model_cols = [
        "positive_share", "audience_mean_partisanship", "author_partisanship",
        "candidate_order", "formality_bias", "positive_ideology_score", "total_votes",
        "author_gender_male", "author_age_30_39", "author_age_40_over",
        "undirected_sentiment", "sentiment_intensity", "toxicity_score"
    ] + bias_columns
    for col in model_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Impute predictors with 0.0 (no bias) if missing
            if col != "positive_share":
                df[col] = df[col].fillna(0.0)

    df["positive_share"] = df["positive_share"].clip(0.0, 1.0)
    df["total_votes"] = df["total_votes"].clip(lower=1)

    before = len(df)
    df = df.dropna(subset=model_cols)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with residual NaN/Inf values")

    logger.info(f"Merged dataset: {len(df)} polls, {len(df.columns)} columns")
    return df


def fit_glm(df: pd.DataFrame):
    """
    Fit a Binomial GLM predicting positive_share, weighted by total_votes.
    """
    formula = (
        "positive_share ~ audience_mean_partisanship + author_partisanship"
        " + candidate_order + formality_bias + positive_ideology_score"
        " + undirected_sentiment + sentiment_intensity + toxicity_score"
        " + bias_confirmation + bias_anchoring + bias_availability"
        " + bias_social_desirability + bias_acquiescence + bias_demand_characteristics"
        " + author_gender_male"
        " + author_age_30_39 + author_age_40_over"
    )

    # For Binomial proportions, pass the number of trials as var_weights
    # This properly scales the variance (larger polls have tighter variance)
    raw_weights = df["total_votes"].values.astype(float)

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=raw_weights,
    )
    result = model.fit()
    return result


def fit_glm_baseline(df: pd.DataFrame):
    """
    Fit a baseline Binomial GLM predicting positive_share without bias markers.
    """
    formula = (
        "positive_share ~ audience_mean_partisanship + author_partisanship"
        " + author_gender_male"
        " + author_age_30_39 + author_age_40_over"
    )

    raw_weights = df["total_votes"].values.astype(float)

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=raw_weights,
    )
    result = model.fit()
    return result


def fit_ols(df: pd.DataFrame):
    """
    Fit an Ordinary Least Squares (OLS) regression predicting positive_share.
    Used as an interpretable standard reference model.
    """
    formula = (
        "positive_share ~ audience_mean_partisanship + author_partisanship"
        " + candidate_order + formality_bias + positive_ideology_score"
        " + undirected_sentiment + sentiment_intensity + toxicity_score"
        " + bias_confirmation + bias_anchoring + bias_availability"
        " + bias_social_desirability + bias_acquiescence + bias_demand_characteristics"
        " + author_gender_male"
        " + author_age_30_39 + author_age_40_over"
    )

    model = smf.ols(
        formula=formula,
        data=df,
    )
    result = model.fit()
    return result


def fit_ols_baseline(df: pd.DataFrame):
    """
    Fit a baseline Ordinary Least Squares (OLS) regression predicting positive_share without bias markers.
    Used as an interpretable standard reference model.
    """
    formula = (
        "positive_share ~ audience_mean_partisanship + author_partisanship"
        " + author_gender_male"
        " + author_age_30_39 + author_age_40_over"
    )

    model = smf.ols(
        formula=formula,
        data=df,
    )
    result = model.fit()
    return result


def build_poststrat_frame(
    df: pd.DataFrame,
    election: str = DEFAULT_ELECTION,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a post-stratification frame crossing partisan × ideological strata.

    reference_df: dataframe used to compute quantile anchors for the partisan
    profiles. Defaults to df itself. Pass the full analysis_df when calling
    from a rolling-window context so that quantile anchors stay stable across
    windows that may have very few observations.
    """
    ecfg = get_election_config(election)
    scwg_cfg = ecfg["scwg"]

    partisan_strata = scwg_cfg["partisan_strata"]
    ideological_strata = scwg_cfg["ideological_strata"]
    partisan_profiles_cfg = scwg_cfg["partisan_profiles"]
    ideology_offsets = scwg_cfg["ideology_offsets"]
    demographic_profiles = scwg_cfg["demographic_profiles"]

    qref = reference_df if reference_df is not None else df

    # When a reference_df is provided (rolling-window mode), the quantile anchors
    # come from the full reference distribution (stable), but are shifted by the
    # window's mean deviation from the global mean.  This preserves the relative
    # partisan structure (Republican > Independent > Democrat) while allowing the
    # absolute level to drift with genuine weekly variation.  Using the mean (not
    # a per-window quantile) keeps the shift low-variance.
    def _mean_shift(col: str) -> float:
        if reference_df is None:
            return 0.0
        w_mean = df[col].mean()
        g_mean = qref[col].mean()
        if pd.isna(w_mean) or pd.isna(g_mean):
            return 0.0
        return float(w_mean - g_mean)

    aud_shift = _mean_shift("audience_mean_partisanship")
    auth_shift = _mean_shift("author_partisanship")
    cons_shift = _mean_shift("positive_ideology_score")

    partisan_profiles = {}
    for p_name, quantiles in partisan_profiles_cfg.items():
        partisan_profiles[p_name] = {
            "aud": qref["audience_mean_partisanship"].quantile(quantiles["aud_quantile"]) + aud_shift,
            "auth": qref["author_partisanship"].quantile(quantiles["auth_quantile"]) + auth_shift,
            "cons": qref["positive_ideology_score"].quantile(quantiles["cons_quantile"]) + cons_shift,
        }

    # Zero-Bias Counterfactual: To isolate true population preference, we must project
    # a perfectly neutral polling environment (0 toxicity, 0 bias, neutral order)
    median_order = 0.0
    median_formality = 0.0
    median_sentiment = 0.0
    median_intensity = 0.0
    median_toxicity = 0.0

    bias_columns = [
        "bias_confirmation", "bias_anchoring", "bias_availability",
        "bias_social_desirability", "bias_acquiescence", "bias_demand_characteristics"
    ]
    bias_medians = {b: 0.0 for b in bias_columns}

    rows = []
    for p_name, p_weight in partisan_strata.items():
        for i_name, i_weight in ideological_strata.items():
            profile = partisan_profiles[p_name]
            demo = demographic_profiles[p_name]
            mult = ideology_offsets[i_name]["cons_mult"]
            rows.append({
                "partisan_stratum": p_name,
                "ideology_stratum": i_name,
                "audience_mean_partisanship": profile["aud"],
                "author_partisanship": profile["auth"],
                "candidate_order": median_order,
                "formality_bias": median_formality,
                "undirected_sentiment": median_sentiment,
                "sentiment_intensity": median_intensity,
                "toxicity_score": median_toxicity,
                **bias_medians,
                "positive_ideology_score": profile["cons"] * mult,
                "author_gender_male": demo["gender_male"],
                "author_age_30_39": demo["age_30_39"],
                "author_age_40_over": demo["age_40_over"],
                "weight": p_weight * i_weight,
            })

    ps_frame = pd.DataFrame(rows)
    logger.info(
        f"Post-stratification frame: {len(ps_frame)} strata, "
        f"total weight = {ps_frame['weight'].sum():.4f}"
    )
    return ps_frame

def poststratify(glm_result, ps_frame: pd.DataFrame) -> float:
    predictions = glm_result.predict(ps_frame)
    return np.average(predictions, weights=ps_frame["weight"])


@app.command()
def main(
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    ecfg = get_election_config(election)
    cand_pos = ecfg["bias_direction"]["positive"]
    actual_pos_share = ecfg["scwg"]["actual_results"][cand_pos]
    market = ecfg["scwg"]["prediction_market"]

    base_df = get_base_dataset(election=election)
    if base_df.empty:
        logger.error("Base dataset is empty — aborting.")
        raise typer.Exit(code=1)

    df = load_and_merge_features(base_df, election=election)
    analysis_df = df.dropna(subset=["positive_share"]).copy()
    logger.info(f"Analysis-ready polls: {len(analysis_df)}")

    if len(analysis_df) < 10:
        logger.error("Too few polls with valid positive_share: aborting.")
        raise typer.Exit(code=1)

    glm_result = fit_glm(analysis_df)
    logger.info("\n" + str(glm_result.summary()))

    ps_frame = build_poststrat_frame(analysis_df, election=election)
    scwg_estimate = poststratify(glm_result, ps_frame)

    ps_frame[f"predicted_{cand_pos.lower()}_share"] = glm_result.predict(ps_frame)
    logger.info("\nPost-stratification predictions per stratum:")
    for _, row in ps_frame.iterrows():
        logger.info(
            f"  {row['partisan_stratum']:>12s} × {row['ideology_stratum']:<14s}  "
            f"weight={row['weight']:.4f}  "
            f"pred_{cand_pos.lower()}={row[f'predicted_{cand_pos.lower()}_share']:.4f}"
        )

    raw_avg = analysis_df["positive_share"].mean()
    weighted_avg = np.average(analysis_df["positive_share"], weights=analysis_df["total_votes"])

    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    logger.info(f"  Raw Unweighted Average ({cand_pos} %) : {raw_avg * 100:.2f}%")
    logger.info(f"  Vote-Weighted Average  ({cand_pos} %) : {weighted_avg * 100:.2f}%")
    logger.info(f"  SCWG Adjusted Estimate  ({cand_pos} %) : {scwg_estimate * 100:.2f}%")
    logger.info(f"  Actual Result          ({cand_pos} %) : {actual_pos_share * 100:.1f}%")

    deviation = (scwg_estimate - actual_pos_share) * 100
    logger.info(f"  Exit-Poll Deviation              : {deviation:+.2f} pp")

    logger.info("")
    logger.info(f"  Prediction Market Reference: {market['source']}")
    for cand in ecfg["candidates"]:
        if cand in market:
            logger.info(f"    {cand} win probability: {market[cand] * 100:.0f}%")
    logger.info(f"    Note: {market.get('note', '')}")

    baseline_glm_result = fit_glm_baseline(analysis_df)
    
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    logger.info("AIC (Akaike Information Criterion): Lower is better. Balances goodness-of-fit and complexity.")
    logger.info(f"  Full GLM AIC:     {glm_result.aic:.2f}")
    logger.info(f"  Baseline GLM AIC: {baseline_glm_result.aic:.2f}")
    
    logger.info("Log-Likelihood: Higher (closer to positive) is better. Measures how well the model explains the data.")
    logger.info(f"  Full GLM Log-Likelihood:     {glm_result.llf:.2f}")
    logger.info(f"  Baseline GLM Log-Likelihood: {baseline_glm_result.llf:.2f}")
    
    if glm_result.aic < baseline_glm_result.aic:
        logger.info("-> Bias markers improved the GLM model (Lower AIC).")
    else:
        logger.info("-> Bias markers did NOT improve the GLM model (Higher or equal AIC).")

    logger.info("")
    logger.info("=" * 70)
    logger.info("STANDARD OLS REFERENCE (Interpretability)")
    logger.info("=" * 70)
    ols_result = fit_ols(analysis_df)
    ols_baseline_result = fit_ols_baseline(analysis_df)
    logger.info("--- Full OLS Model Summary ---")
    logger.info("\n" + str(ols_result.summary()))

    logger.info("--- Variance Inflation Factor (VIF) ---")
    exog = ols_result.model.exog
    exog_names = ols_result.model.exog_names
    
    vif = pd.DataFrame()
    vif["Variable"] = exog_names
    vif["VIF"] = [variance_inflation_factor(exog, i) for i in range(exog.shape[1])]
    
    vif = vif.sort_values("VIF", ascending=False).reset_index(drop=True)
    for idx, row in vif.iterrows():
        vif_val = row["VIF"]
        var_name = row["Variable"]
        if var_name == "Intercept":
            continue
        warning = " (HIGH)" if vif_val > 5.0 else ""
        logger.info(f"  {var_name:<35}: {vif_val:>6.2f}{warning}")
    logger.info("---------------------------------------")

    logger.info("--- Baseline OLS Model Summary ---")
    logger.info("\n" + str(ols_baseline_result.summary()))

    logger.info("=" * 70)
    logger.info("OLS MODEL COMPARISON")
    logger.info("=" * 70)
    logger.info("Adjusted R-squared: Higher is better. Measures variance explained, penalized for extra predictors.")
    logger.info(f"  Full OLS Adj. R-squared:     {ols_result.rsquared_adj:.4f}")
    logger.info(f"  Baseline OLS Adj. R-squared: {ols_baseline_result.rsquared_adj:.4f}")
    
    logger.info("AIC: Lower is better.")
    logger.info(f"  Full OLS AIC:     {ols_result.aic:.2f}")
    logger.info(f"  Baseline OLS AIC: {ols_baseline_result.aic:.2f}")

    if ols_result.rsquared_adj > ols_baseline_result.rsquared_adj and ols_result.aic < ols_baseline_result.aic:
        logger.info("-> Bias markers improved the OLS model (Higher Adj. R-squared and Lower AIC).")
    elif ols_result.rsquared_adj > ols_baseline_result.rsquared_adj:
        logger.info("-> Bias markers improved the OLS model's Adj. R-squared, but not AIC.")
    elif ols_result.aic < ols_baseline_result.aic:
        logger.info("-> Bias markers improved the OLS model's AIC, but not Adj. R-squared.")
    else:
        logger.info("-> Bias markers did NOT improve the OLS model.")

    logger.success("SCWG pipeline completed successfully.")


if __name__ == "__main__":
    app()
