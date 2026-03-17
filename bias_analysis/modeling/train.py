"""
Multilevel Regression and Post-stratification (MRP) for Twitter Poll Bias Adjustment.

This module implements a frequentist MRP pipeline that:
1. Merges raw poll data with project bias markers (candidate order, formality, political leaning).
2. Fits a Binomial GLM weighted by poll size.
3. Post-stratifies predictions over 2020 National Exit Poll population strata.
4. Benchmarks the adjusted estimate against the actual 2020 election result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from loguru import logger
from tqdm import tqdm
import typer

from bias_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR
from bias_analysis.dataset import get_base_dataset

app = typer.Typer()

# ---------------------------------------------------------------------------
# Actual 2020 election results (% of popular vote)
# ---------------------------------------------------------------------------
ACTUAL_TRUMP_SHARE = 0.468  # 46.8%
ACTUAL_BIDEN_SHARE = 0.513  # 51.3%

# ---------------------------------------------------------------------------
# 2020 National Exit Poll proportions
# ---------------------------------------------------------------------------
PARTISAN_STRATA = {"Republican": 0.36, "Democrat": 0.37, "Independent": 0.26}
IDEOLOGICAL_STRATA = {"Conservative": 0.38, "Moderate": 0.38, "Liberal": 0.24}

# 2020 National Exit Poll gender split (48% male voters)
GENDER_MALE_SHARE = 0.48

# 2020 National Exit Poll age split (mapped to M3 brackets)
# M3 brackets: <=18, 19-29, 30-39, >=40
# Exit poll: 18-29 ~17%, 30-44 ~23%, 45-64 ~28%, 65+ ~32%
AGE_19_29_SHARE = 0.17
AGE_30_39_SHARE = 0.23
AGE_40_OVER_SHARE = 0.60


def load_and_merge_features(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the three processed feature CSVs into the base poll dataframe.

    Engineered columns added:
        - candidate_order  : (Biden_position − Trump_position) × 0.25
        - formality_bias   : trump_formal − biden_formal  (formal=1, informal=−1)
        - conservative_score: from political_leaning_features.csv
        - trump_share      : Trump_percentage / 100
    """
    df = base_df.copy()
    df["tweet_id"] = df["tweet_id"].astype(str)

    co_path = PROCESSED_DATA_DIR / "candidate_order_features.csv"
    if co_path.exists():
        co_df = pd.read_csv(co_path)
        co_df["poll_id"] = co_df["poll_id"].astype(str)
        df = df.merge(
            co_df[["poll_id", "Trump_position", "Biden_position",
                   "Trump_percentage", "Biden_percentage", "Trump_votes",
                   "Biden_votes"]],
            left_on="tweet_id", right_on="poll_id", how="left",
        )
        df["candidate_order"] = (
            (df["Biden_position"] - df["Trump_position"]) * 0.25
        )
        missing_mask = df["Biden_position"].isna() | df["Trump_position"].isna()
        df.loc[missing_mask, "candidate_order"] = np.nan
    else:
        logger.warning(f"Missing {co_path.name} — candidate_order will be NaN")
        for col in ["Trump_percentage", "Biden_percentage",
                     "Trump_votes", "Biden_votes"]:
            df[col] = np.nan
        df["candidate_order"] = np.nan

    fi_path = PROCESSED_DATA_DIR / "formal_informal_appellatives.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        fi_df["poll_id"] = fi_df["poll_id"].astype(str)
        df = df.merge(
            fi_df[["poll_id", "Trump_label", "Biden_label"]],
            left_on="tweet_id", right_on="poll_id", how="left",
            suffixes=("", "_fi"),
        )
        label_map = {"formal": 1.0, "informal": -1.0, "neutral": 0.0}
        trump_formal = df["Trump_label"].map(label_map).fillna(0.0)
        biden_formal = df["Biden_label"].map(label_map).fillna(0.0)
        df["formality_bias"] = trump_formal - biden_formal
    else:
        logger.warning(f"Missing {fi_path.name} — formality_bias will be 0")
        df["formality_bias"] = 0.0

    pl_path = PROCESSED_DATA_DIR / "political_leaning_features.csv"
    if pl_path.exists():
        pl_df = pd.read_csv(pl_path)
        pl_df["poll_id"] = pl_df["poll_id"].astype(str)
        df = df.merge(
            pl_df[["poll_id", "conservative_score"]],
            left_on="tweet_id", right_on="poll_id", how="left",
            suffixes=("", "_pl"),
        )
    else:
        logger.warning(f"Missing {pl_path.name} — conservative_score will be 0")
        df["conservative_score"] = 0.0

    # --- Target: trump_share ------------------------------------------------
    df["trump_share"] = df["Trump_percentage"] / 100.0

    for col in ["audience_mean_partisanship", "author_partisanship"]:
        if col in df.columns:
            col_mean = df[col].mean()
            fill_value = col_mean if pd.notna(col_mean) else 0.0
            df[col] = df[col].fillna(fill_value)
        else:
            df[col] = 0.0

    for col in ["candidate_order", "formality_bias", "conservative_score"]:
        df[col] = df[col].fillna(0.0)

    # Fill demographic probabilities with column mean (or 0.5 neutral prior)
    for col in ["author_gender_male", "author_age_19_29", "author_age_30_39", "author_age_40_over"]:
        if col in df.columns:
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean if pd.notna(col_mean) else 0.5)
        else:
            df[col] = 0.5

    df["trump_share"] = df["trump_share"].fillna(df["trump_share"].median())

    model_cols = [
        "trump_share", "audience_mean_partisanship", "author_partisanship",
        "candidate_order", "formality_bias", "conservative_score", "total_votes",
        "author_gender_male", "author_age_19_29", "author_age_30_39", "author_age_40_over",
    ]
    for col in model_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["trump_share"] = df["trump_share"].clip(0.0, 1.0)
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
    Fit a Binomial GLM predicting trump_share, weighted by total_votes.

    Returns:
        statsmodels GLMResultsWrapper
    """
    formula = (
        "trump_share ~ audience_mean_partisanship + author_partisanship"
        " + candidate_order + formality_bias + conservative_score"
        " + author_gender_male"
        " + author_age_19_29 + author_age_30_39 + author_age_40_over"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total_votes"].values.astype(float),
    )
    result = model.fit()
    return result


def build_poststrat_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a post-stratification frame crossing partisan × ideological strata.

    Characteristic feature values for each stratum are derived from the
    empirical distribution of the modelling dataset:
        - Republican / Conservative  → 75th-pctile partisanship, high conserv. score
        - Democrat  / Liberal        → 25th-pctile partisanship, low conserv. score
        - Independent / Moderate     → median values

    Returns:
        DataFrame with one row per stratum and columns matching the GLM formula
        plus a ``weight`` column (joint population proportion).
    """
    aud_q25 = df["audience_mean_partisanship"].quantile(0.25)
    aud_q50 = df["audience_mean_partisanship"].quantile(0.50)
    aud_q75 = df["audience_mean_partisanship"].quantile(0.75)

    auth_q25 = df["author_partisanship"].quantile(0.25)
    auth_q50 = df["author_partisanship"].quantile(0.50)
    auth_q75 = df["author_partisanship"].quantile(0.75)

    cons_q25 = df["conservative_score"].quantile(0.25)
    cons_q50 = df["conservative_score"].quantile(0.50)
    cons_q75 = df["conservative_score"].quantile(0.75)

    median_order = df["candidate_order"].median()
    median_formality = df["formality_bias"].median()

    partisan_profiles = {
        "Republican":   {"aud": aud_q75, "auth": auth_q75, "cons": cons_q75},
        "Democrat":     {"aud": aud_q25, "auth": auth_q25, "cons": cons_q25},
        "Independent":  {"aud": aud_q50, "auth": auth_q50, "cons": cons_q50},
    }

    ideology_offsets = {
        "Conservative": {"cons_mult": 1.25},
        "Moderate":     {"cons_mult": 1.00},
        "Liberal":      {"cons_mult": 0.75},
    }

    # Demographic profiles per partisan stratum (from exit polls)
    # Republicans skew older and more male; Democrats skew younger and more female
    demographic_profiles = {
        "Republican":  {"gender_male": 0.52, "age_19_29": 0.11, "age_30_39": 0.19, "age_40_over": 0.70},
        "Democrat":    {"gender_male": 0.43, "age_19_29": 0.24, "age_30_39": 0.27, "age_40_over": 0.49},
        "Independent": {"gender_male": GENDER_MALE_SHARE, "age_19_29": AGE_19_29_SHARE, "age_30_39": AGE_30_39_SHARE, "age_40_over": AGE_40_OVER_SHARE},
    }

    rows = []
    for p_name, p_weight in PARTISAN_STRATA.items():
        for i_name, i_weight in IDEOLOGICAL_STRATA.items():
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
                "conservative_score": profile["cons"] * mult,
                "author_gender_male": demo["gender_male"],
                "author_age_19_29": demo["age_19_29"],
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
    """
    Compute the MRP estimate: weighted average of stratum-level predictions.
    """
    predictions = glm_result.predict(ps_frame)
    mrp_estimate = np.average(predictions, weights=ps_frame["weight"])
    return mrp_estimate


def get_prediction_market_price() -> dict:
    """
    Placeholder for prediction-market benchmark.

    In live usage this would query Betfair / PredictIt historical API.
    Prices are quoted in cents where 1¢ = 1% implied probability.

    Returns:
        dict with market-implied probabilities (as fractions 0-1).
    """
    # Representative closing prices from prediction markets circa Nov 2 2020
    return {
        "source": "PredictIt (placeholder)",
        "trump_win_prob": 0.39,   # ~39¢ contract
        "biden_win_prob": 0.63,   # ~63¢ contract
        "note": "These are win-probability prices, not vote-share estimates.",
    }


@app.command()
def main():
    """
    Run the full MRP pipeline: load → merge → GLM → post-stratify → benchmark.
    """
    base_df = get_base_dataset()
    if base_df.empty:
        logger.error("Base dataset is empty — aborting.")
        raise typer.Exit(code=1)

    df = load_and_merge_features(base_df)

    analysis_df = df.dropna(subset=["trump_share"]).copy()
    logger.info(f"Analysis-ready polls: {len(analysis_df)}")

    if len(analysis_df) < 10:
        logger.error("Too few polls with valid trump_share: aborting.")
        raise typer.Exit(code=1)

    glm_result = fit_glm(analysis_df)
    logger.info("\n" + str(glm_result.summary()))

    ps_frame = build_poststrat_frame(analysis_df)
    mrp_estimate = poststratify(glm_result, ps_frame)

    ps_frame["predicted_trump_share"] = glm_result.predict(ps_frame)
    logger.info("\nPost-stratification predictions per stratum:")
    for _, row in ps_frame.iterrows():
        logger.info(
            f"  {row['partisan_stratum']:>12s} × {row['ideology_stratum']:<14s}  "
            f"weight={row['weight']:.4f}  "
            f"pred_trump={row['predicted_trump_share']:.4f}"
        )

    raw_avg = analysis_df["trump_share"].mean()
    weighted_avg = np.average(
        analysis_df["trump_share"],
        weights=analysis_df["total_votes"],
    )

    market = get_prediction_market_price()

    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    logger.info(f"  Raw Unweighted Average (Trump %) : {raw_avg * 100:.2f}%")
    logger.info(f"  Vote-Weighted Average  (Trump %) : {weighted_avg * 100:.2f}%")
    logger.info(f"  MRP Adjusted Estimate  (Trump %) : {mrp_estimate * 100:.2f}%")
    logger.info(f"  Actual 2020 Result     (Trump %) : {ACTUAL_TRUMP_SHARE * 100:.1f}%")

    deviation = (mrp_estimate - ACTUAL_TRUMP_SHARE) * 100
    logger.info(f"  Exit-Poll Deviation              : {deviation:+.2f} pp")

    logger.info("")
    logger.info(f"  Prediction Market Reference: {market['source']}")
    logger.info(f"    Trump win probability: {market['trump_win_prob'] * 100:.0f}%")
    logger.info(f"    Biden win probability: {market['biden_win_prob'] * 100:.0f}%")
    logger.info(f"    Note: {market['note']}")

    logger.success("MRP pipeline completed successfully.")


if __name__ == "__main__":
    app()
