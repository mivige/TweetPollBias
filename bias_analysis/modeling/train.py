"""
Multilevel Regression and Post-stratification (MRP) for Twitter Poll Bias Adjustment.

This module implements a frequentist MRP pipeline that:
1. Merges raw poll data with project bias markers (candidate order, formality, political leaning).
2. Fits a Binomial GLM weighted by poll size.
3. Post-stratifies predictions over population strata.
4. Benchmarks the adjusted estimate against the actual election result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

    # --- Target: positive_share ---------------------------------------------
    df["positive_share"] = df[f"{cand_pos}_percentage"] / 100.0

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

    df["positive_share"] = df["positive_share"].fillna(df["positive_share"].median())

    model_cols = [
        "positive_share", "audience_mean_partisanship", "author_partisanship",
        "candidate_order", "formality_bias", "positive_ideology_score", "total_votes",
        "author_gender_male", "author_age_30_39", "author_age_40_over",
    ]
    for col in model_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
        " + author_gender_male"
        " + author_age_30_39 + author_age_40_over"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total_votes"].values.astype(float),
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

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total_votes"].values.astype(float),
    )
    result = model.fit()
    return result


def build_poststrat_frame(df: pd.DataFrame, election: str = DEFAULT_ELECTION) -> pd.DataFrame:
    """
    Build a post-stratification frame crossing partisan × ideological strata.
    """
    ecfg = get_election_config(election)
    mrp_cfg = ecfg["mrp"]
    
    partisan_strata = mrp_cfg["partisan_strata"]
    ideological_strata = mrp_cfg["ideological_strata"]
    partisan_profiles_cfg = mrp_cfg["partisan_profiles"]
    ideology_offsets = mrp_cfg["ideology_offsets"]
    demographic_profiles = mrp_cfg["demographic_profiles"]

    partisan_profiles = {}
    for p_name, quantiles in partisan_profiles_cfg.items():
        partisan_profiles[p_name] = {
            "aud": df["audience_mean_partisanship"].quantile(quantiles["aud_quantile"]),
            "auth": df["author_partisanship"].quantile(quantiles["auth_quantile"]),
            "cons": df["positive_ideology_score"].quantile(quantiles["cons_quantile"]),
        }

    median_order = df["candidate_order"].median()
    median_formality = df["formality_bias"].median()

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
    actual_pos_share = ecfg["mrp"]["actual_results"][cand_pos]
    market = ecfg["mrp"]["prediction_market"]

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
    mrp_estimate = poststratify(glm_result, ps_frame)

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
    logger.info(f"  MRP Adjusted Estimate  ({cand_pos} %) : {mrp_estimate * 100:.2f}%")
    logger.info(f"  Actual Result          ({cand_pos} %) : {actual_pos_share * 100:.1f}%")

    deviation = (mrp_estimate - actual_pos_share) * 100
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


    logger.success("MRP pipeline completed successfully.")


if __name__ == "__main__":
    app()
