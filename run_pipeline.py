"""
End-to-End Pipeline Runner for TweetPollBias

This script executes the entire data pipeline for a specified election:
1. Feature extraction (candidate order, formality, leaning, correlations)
2. MRP Modeling (GLM fitting, post-stratification)
3. Visualizations (Plots and MRP dashboard)
"""

import sys
import typer
from loguru import logger

from bias_analysis.election_configs import get_election_config

# Import pipeline steps
from bias_analysis import features
from bias_analysis.modeling import train
from bias_analysis import plots

app = typer.Typer(help="Run the complete TweetPollBias pipeline.")

@app.command()
def main(
    election: str = typer.Option("us20", help="Election code (e.g. 'us20', 'us16')"),
    skip_features: bool = typer.Option(False, "--skip-features", help="Skip feature extraction step"),
    skip_modeling: bool = typer.Option(False, "--skip-modeling", help="Skip MRP modeling step"),
    skip_plots: bool = typer.Option(False, "--skip-plots", help="Skip visualization step"),
    max_rows: int = typer.Option(None, help="Limit number of rows processed per file (useful for testing)"),
):
    """
    Run the full analysis pipeline sequentially for a given election.
    """
    try:
        cfg = get_election_config(election)
    except ValueError as e:
        logger.error(e)
        raise typer.Exit(code=1)
        
    logger.info("=" * 80)
    logger.info(f"STARTING PIPELINE FOR ELECTION: {cfg.get('display_name', election)}")
    logger.info("=" * 80)

    # 1. Feature Extraction
    if not skip_features:
        logger.info("\n=== PHASE 1: FEATURE EXTRACTION ===")
        features.candidate_order(election=election)
        features.political_leaning(election=election, max_samples=max_rows)
        features.formal_vs_informal(election=election)
        features.pearson_correlation(election=election, max_rows=max_rows)
    else:
        logger.warning("\n=== PHASE 1: FEATURE EXTRACTION (SKIPPED) ===")

    # 2. MRP Modeling
    if not skip_modeling:
        logger.info("\n=== PHASE 2: MRP MODELING ===")
        train.main(election=election)
    else:
        logger.warning("\n=== PHASE 2: MRP MODELING (SKIPPED) ===")
        
    # 3. Visualizations
    if not skip_plots:
        logger.info("\n=== PHASE 3: VISUALIZATIONS AND DASHBOARD ===")
        plots.run_all(election=election)
    else:
        logger.warning("\n=== PHASE 3: VISUALIZATIONS (SKIPPED) ===")

    logger.success("=" * 80)
    logger.success(f"PIPELINE COMPLETED SUCCESSFULLY FOR: {cfg.get('display_name', election)}")
    logger.success("=" * 80)


if __name__ == "__main__":
    app()
