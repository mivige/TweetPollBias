"""
Visualization module for Twitter poll bias analysis.

This module creates publication-quality visualizations to analyze position bias
in Twitter polls. All election-specific references (candidate names, colors,
labels, milestones) are loaded from bias_analysis.election_configs so the same
plotting pipeline works across different elections.
"""

from pathlib import Path
from datetime import datetime, timedelta

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from bias_analysis.config import get_election_paths
from bias_analysis.election_configs import get_election_config
from bias_analysis.dataset import load_predictit_data

app = typer.Typer()

# Default election code used when --election is not provided on the CLI.
DEFAULT_ELECTION = "us20"


@app.command()
def candidate_order(
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    """
    Generate advanced scatter plots analyzing candidate position bias in Twitter polls.

    Creates one panel per candidate showing how poll position correlates with vote percentage.
    Saves both the plot and detailed statistics to files.
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    candidates = ecfg["candidates"]
    colors = ecfg["candidate_colors"]

    input_path = paths.processed_dir / "candidate_order_features.csv"
    output_path = paths.figures_dir / "candidate_order_plot.png"

    # Load processed candidate order features
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} poll records")

    # Filter to polls containing at least two valid candidates
    candidate_cols = [f'{c}_position' for c in candidates if f'{c}_position' in df.columns]
    filter_mask = df[candidate_cols].notna().sum(axis=1) >= 2
    head_to_head = df[filter_mask].copy()

    if len(head_to_head) == 0:
        logger.error(f"No head-to-head polls found for candidates: {candidates}")
        return

    logger.info(f"Found {len(head_to_head)} head-to-head polls for analysis")

    def improved_scatter(df_candidate, title, ax, color):
        """
        Create sophisticated scatter plot with multiple statistical layers.

        Args:
            df_candidate: DataFrame with candidate position and percentage data
            title: Plot title
            ax: Matplotlib axis object
            color: Scatter point color

        Returns:
            Modified axis object
        """

        # Standardize column names for consistent processing
        candidate_name = df_candidate['candidate'].iloc[0]
        plot_df = df_candidate.rename(columns={
            f'{candidate_name}_position': 'position',
            f'{candidate_name}_percentage': 'percentage'
        }).copy()

        # Remove zero-percentage polls to focus on competitive races
        plot_df_nonzero = plot_df[plot_df['percentage'] > 0].copy()

        if len(plot_df_nonzero) == 0:
            logger.warning(f"No non-zero data for {candidate_name}")
            return ax

        # Add horizontal jitter
        x_base = plot_df_nonzero['position'].values.astype(float)
        jitter = np.random.normal(loc=0, scale=0.08, size=len(plot_df_nonzero))
        x_jittered = x_base + jitter

        # Background violin plots
        try:
            sns.violinplot(x='position', y='percentage', data=plot_df_nonzero,
                          inner=None, color='lightgray', cut=0, linewidth=0,
                          alpha=0.3, zorder=0, ax=ax)
        except Exception as e:
            logger.warning(f"Could not create violin plot for {candidate_name}: {e}")

        # Main scatter plot with candidate-specific colors
        ax.scatter(
            x_jittered, plot_df_nonzero['percentage'],
            alpha=0.6, s=36, edgecolor='white', linewidth=0.4,
            c=color,
            zorder=3
        )

        grp = plot_df_nonzero.groupby('position')['percentage']
        medians = grp.median()
        means = grp.mean()
        sems = grp.sem().fillna(0)

        x_positions = sorted(plot_df_nonzero['position'].unique())

        if len(x_positions) > 0:
            ax.errorbar(x_positions, means.loc[x_positions], yerr=sems.loc[x_positions],
                       fmt='D', color='black', markersize=7, capsize=5,
                       label='Mean ± SEM', zorder=4)
            for pos in x_positions:
                if pos in medians.index:
                    ax.plot(pos, medians.loc[pos], marker='s', color='darkgreen',
                           markersize=8, zorder=4)

        overall_median = plot_df_nonzero['percentage'].median()
        ax.axhline(overall_median, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlabel('Poll Position')
        ax.set_ylabel('Vote Percentage (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        max_position = plot_df_nonzero['position'].max()
        min_position = plot_df_nonzero['position'].min()
        ax.set_xlim(min_position - 0.5, max_position + 0.5)

        all_positions = sorted(plot_df_nonzero['position'].unique())
        ax.set_xticks(all_positions)
        ax.set_xticklabels([str(int(pos)) for pos in all_positions])

        ax.set_ylim(-2, 102)

        return ax

    # Setup publication-quality plotting style
    sns.set_style("whitegrid")
    n_candidates = len(candidates)
    fig, axes = plt.subplots(1, n_candidates, figsize=(8 * n_candidates, 7))
    if n_candidates == 1:
        axes = [axes]

    # Prepare candidate-specific datasets
    plot_data = []

    for candidate in candidates:
        candidate_data = head_to_head[[f'{candidate}_position', f'{candidate}_percentage']].dropna()

        if len(candidate_data) == 0:
            continue

        candidate_data['candidate'] = candidate
        plot_data.append(candidate_data)

    # Generate multi-panel visualization
    for i, (candidate, cand_data) in enumerate(zip(candidates, plot_data)):
        color = colors.get(candidate, 'tab:gray')
        improved_scatter(cand_data, f"{candidate} — Position vs Vote Percentage", axes[i], color)

    # Shared legend explaining statistical markers
    if len(plot_data) >= 2:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='black', linestyle='None',
                   markersize=7, label='Mean ± SEM'),
            Line2D([0], [0], marker='s', color='darkgreen', linestyle='None',
                   markersize=8, label='Median'),
            Line2D([0], [0], color='gray', linestyle='--', label='Overall Median')
        ]

        # Position legend below plots for clean layout
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                  frameon=True, fancybox=True, shadow=True, ncol=3)

    # Save high-resolution plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Plot saved to {output_path}")
    plt.close()

    # Generate detailed statistical summary for reference
    stats_output_path = paths.reports_dir / "candidate_order_statistics.txt"
    with open(stats_output_path, 'w') as f:
        f.write("=== Candidate Order Analysis - Summary Statistics ===\n\n")

        for i, candidate in enumerate(candidates):
            if i < len(plot_data):
                candidate_data = plot_data[i]
                candidate_nonzero = candidate_data[candidate_data[f'{candidate}_percentage'] > 0]

                f.write(f"{candidate}:\n")
                f.write(f"  Total polls: {len(candidate_data)}\n")
                f.write(f"  Non-zero polls: {len(candidate_nonzero)}\n")

                if len(candidate_nonzero) > 0:
                    # Position-specific performance metrics
                    for pos in sorted(candidate_nonzero[f'{candidate}_position'].unique()):
                        pos_data = candidate_nonzero[candidate_nonzero[f'{candidate}_position'] == pos]
                        mean_perf = pos_data[f'{candidate}_percentage'].mean()
                        median_perf = pos_data[f'{candidate}_percentage'].median()
                        count = len(pos_data)
                        f.write(f"  Position {pos}: {count} polls, Mean: {mean_perf:.1f}%, Median: {median_perf:.1f}%\n")

                    # Overall performance summary
                    overall_mean = candidate_nonzero[f'{candidate}_percentage'].mean()
                    overall_median = candidate_nonzero[f'{candidate}_percentage'].median()
                    f.write(f"  Overall: Mean: {overall_mean:.1f}%, Median: {overall_median:.1f}%\n")

                f.write("\n")

    logger.success(f"Summary statistics saved to {stats_output_path}")

    return plot_data

@app.command()
def appellatives(
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    """
    Generate advanced scatter plots analyzing formality bias in Twitter poll appellatives.

    Creates one panel per candidate showing how appellative formality correlates with vote percentage.
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    candidates = ecfg["candidates"]
    colors = ecfg["candidate_colors"]

    input_path = paths.processed_dir / "formal_informal_appellatives.csv"
    output_path = paths.figures_dir / "formal_informal_appellatives_plot.png"

    # Load processed appellative features
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} poll records")

    # Support both new formality_score columns and legacy _label columns
    _score_col_exists = any(f'{c}_formality_score' in df.columns for c in candidates)
    _label_col_exists = any(f'{c}_label' in df.columns for c in candidates)

    if _score_col_exists:
        formality_col_suffix  = '_formality_score'
        use_score_mode        = True
        # 6-level ordinal scale (van den Berg 2019 + extensions)
        formality_order       = ['ADJ_PET_NAME', 'FN', 'LN', 'FNLN', 'TLN', 'TFNLN']
        formality_score_map   = {cat: i for i, cat in enumerate(formality_order)}
        formality_x_labels    = ['Neg Adj/Pet Name', 'First', 'Last',
                                  'First+Last', 'Title+Last', 'Title+First+Last']
    else:
        formality_col_suffix  = '_label'
        use_score_mode        = False
        formality_order       = ['informal', 'neutral', 'formal']
        formality_score_map   = {label: i for i, label in enumerate(formality_order)}
        formality_x_labels    = [l.capitalize() for l in formality_order]

    # Filter to polls containing at least two candidates with formality data
    if use_score_mode:
        candidate_cols = [f'{c}_formality_score' for c in candidates if f'{c}_formality_score' in df.columns]
    else:
        candidate_cols = [f'{c}_label' for c in candidates if f'{c}_label' in df.columns]
    filter_mask = df[candidate_cols].notna().sum(axis=1) >= 2
    head_to_head = df[filter_mask].copy()

    if len(head_to_head) == 0:
        logger.error(f"No head-to-head polls with formality data found for: {candidates}")
        return

    logger.info(f"Found {len(head_to_head)} head-to-head polls with appellative data")

    def improved_formality_scatter(df_candidate, title, ax, color):
        """
        Create scatter plot showing formality category vs vote percentage.
        Works with both the new 7-level formality_score columns and the
        legacy 3-level _label columns.
        """
        candidate_name = df_candidate['candidate'].iloc[0]

        if use_score_mode:
            # Map numeric score to category name for grouping
            score_col = f'{candidate_name}_formality_score'
            cat_col   = f'{candidate_name}_formality_category'
            pct_col   = f'{candidate_name}_percentage'
            # Build a 'formality' column from category if available, else infer from score
            if cat_col in df_candidate.columns:
                plot_df = df_candidate.rename(columns={cat_col: 'formality', pct_col: 'percentage'}).copy()
            else:
                # Reverse-map score to label
                score_to_cat = {v: k for k, v in formality_score_map.items()}
                plot_df = df_candidate.copy()
                plot_df['formality'] = plot_df[score_col].map(score_to_cat)
                plot_df = plot_df.rename(columns={pct_col: 'percentage'})
        else:
            plot_df = df_candidate.rename(columns={
                f'{candidate_name}_label': 'formality',
                f'{candidate_name}_percentage': 'percentage',
            }).copy()

        plot_df_nonzero = plot_df[plot_df['percentage'] > 0].copy()
        if len(plot_df_nonzero) == 0:
            logger.warning(f"No non-zero data for {candidate_name}")
            return ax

        available_formalities = [f for f in formality_order if f in plot_df_nonzero['formality'].values]
        if not available_formalities:
            logger.warning(f"No formality data available for {candidate_name}")
            return ax

        plot_df_nonzero['formality_numeric'] = plot_df_nonzero['formality'].map(formality_score_map)
        plot_df_nonzero = plot_df_nonzero.dropna(subset=['formality_numeric'])

        # Add horizontal jitter to scatter points for better visibility
        x_base = plot_df_nonzero['formality_numeric'].values.astype(float)
        jitter = np.random.normal(loc=0, scale=0.08, size=len(plot_df_nonzero))
        x_jittered = x_base + jitter

        # Background violin plots
        try:
            violin_data = plot_df_nonzero.copy()
            sns.violinplot(x='formality', y='percentage', data=violin_data,
                          order=available_formalities, inner=None, color='lightgray',
                          cut=0, linewidth=0, alpha=0.3, zorder=0, ax=ax)
        except Exception as e:
            logger.warning(f"Could not create violin plot for {candidate_name}: {e}")

        # Main scatter plot with candidate-specific colors
        ax.scatter(
            x_jittered, plot_df_nonzero['percentage'],
            alpha=0.6, s=36, edgecolor='white', linewidth=0.4, c=color, zorder=3
        )

        # Calculate formality-wise statistics for overlays
        grp = plot_df_nonzero.groupby('formality_numeric')['percentage']
        medians = grp.median()
        means   = grp.mean()
        sems    = grp.sem().fillna(0)

        x_positions = sorted(plot_df_nonzero['formality_numeric'].unique())

        if len(x_positions) > 0:
            ax.errorbar(x_positions, means.loc[x_positions], yerr=sems.loc[x_positions],
                       fmt='D', color='black', markersize=7, capsize=5,
                       label='Mean ± SEM', zorder=4)
            for pos in x_positions:
                if pos in medians.index:
                    ax.plot(pos, medians.loc[pos], marker='s', color='darkgreen',
                           markersize=8, zorder=4)

        overall_median = plot_df_nonzero['percentage'].median()
        ax.axhline(overall_median, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlabel('Appellative Formality')
        ax.set_ylabel('Vote Percentage (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(-0.5, len(formality_order) - 0.5)
        ax.set_xticks(range(len(formality_order)))
        ax.set_xticklabels(formality_x_labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(-2, 102)

        return ax

    # Setup publication-quality plotting style
    sns.set_style("whitegrid")
    n_candidates = len(candidates)
    fig, axes = plt.subplots(1, n_candidates, figsize=(8 * n_candidates, 7))
    if n_candidates == 1:
        axes = [axes]

    # Prepare candidate-specific datasets
    plot_data = []

    for candidate in candidates:
        if use_score_mode:
            cat_col   = f'{candidate}_formality_category'
            score_col = f'{candidate}_formality_score'
            pct_col   = f'{candidate}_percentage'
            avail_cols = [c for c in [cat_col, score_col, pct_col] if c in head_to_head.columns]
            candidate_data = head_to_head[avail_cols].dropna(subset=[pct_col])
        else:
            label_col = f'{candidate}_label'
            pct_col   = f'{candidate}_percentage'
            candidate_data = head_to_head[[label_col, pct_col]].dropna()

        if len(candidate_data) == 0:
            continue
        candidate_data = candidate_data.copy()
        candidate_data['candidate'] = candidate
        plot_data.append(candidate_data)

    # Generate multi-panel visualization
    for i, (candidate, cand_data) in enumerate(zip(candidates, plot_data)):
        color = colors.get(candidate, 'tab:gray')
        improved_formality_scatter(cand_data, f"{candidate} — Appellative Formality vs Vote Percentage", axes[i], color)

    # Shared legend explaining statistical markers
    if len(plot_data) >= 2:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='black', linestyle='None',
                   markersize=7, label='Mean ± SEM'),
            Line2D([0], [0], marker='s', color='darkgreen', linestyle='None',
                   markersize=8, label='Median'),
            Line2D([0], [0], color='gray', linestyle='--', label='Overall Median')
        ]

        # Position legend below plots for clean layout
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                  frameon=True, fancybox=True, shadow=True, ncol=3)

    # Save high-resolution plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Plot saved to {output_path}")
    plt.close()

    # Generate detailed statistical summary for reference
    stats_output_path = paths.reports_dir / "formal_informal_appellatives_statistics.txt"
    with open(stats_output_path, 'w') as f:
        f.write("=== Formal vs Informal Appellatives Analysis - Summary Statistics ===\n\n")

        for i, candidate in enumerate(candidates):
            if i < len(plot_data):
                candidate_data = plot_data[i]
                candidate_nonzero = candidate_data[candidate_data[f'{candidate}_percentage'] > 0]

                f.write(f"{candidate}:\n")
                f.write(f"  Total polls: {len(candidate_data)}\n")
                f.write(f"  Non-zero polls: {len(candidate_nonzero)}\n")

                if len(candidate_nonzero) > 0:
                    # Category-specific performance metrics
                    for formality in formality_order:
                        cat_col = f'{candidate}_formality_category' if use_score_mode else f'{candidate}_label'
                        if cat_col in candidate_nonzero.columns:
                            formality_data = candidate_nonzero[candidate_nonzero[cat_col] == formality]
                        else:
                            formality_data = pd.DataFrame()
                        if len(formality_data) > 0:
                            mean_perf   = formality_data[f'{candidate}_percentage'].mean()
                            median_perf = formality_data[f'{candidate}_percentage'].median()
                            count = len(formality_data)
                            f.write(f"  {formality}: {count} polls, Mean: {mean_perf:.1f}%, Median: {median_perf:.1f}%\n")

                    # Overall performance summary
                    overall_mean = candidate_nonzero[f'{candidate}_percentage'].mean()
                    overall_median = candidate_nonzero[f'{candidate}_percentage'].median()
                    f.write(f"  Overall: Mean: {overall_mean:.1f}%, Median: {overall_median:.1f}%\n")

                f.write("\n")

    logger.success(f"Summary statistics saved to {stats_output_path}")

    return plot_data

@app.command()
def leaning(
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    """
    Generate advanced scatter plots analyzing political leaning bias in Twitter polls.

    Creates one panel per candidate showing how political leaning correlates with vote percentage.
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    candidates = ecfg["candidates"]
    colors = ecfg["candidate_colors"]
    column_mapping = ecfg["hypothesis_column_mapping"]
    bias_dir = ecfg["bias_direction"]
    positive_candidate = bias_dir["positive"]
    negative_candidate = bias_dir["negative"]

    input_path = paths.processed_dir / "political_leaning_features.csv"
    output_path = paths.figures_dir / "political_leaning_plot.png"

    # Load processed political leaning features
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} poll records")

    # Resolve column names from config
    cand_A_support_col = column_mapping.get("candidate_A_support", "candidate_A_support_score")
    cand_B_support_col = column_mapping.get("candidate_B_support", "candidate_B_support_score")
    cand_A_oppose_col = column_mapping.get("candidate_A_oppose", "candidate_A_oppose_score")
    cand_B_oppose_col = column_mapping.get("candidate_B_oppose", "candidate_B_oppose_score")
    pos_ideology_col = column_mapping.get("positive_ideology", "positive_ideology_score")
    neg_ideology_col = column_mapping.get("negative_ideology", "negative_ideology_score")

    # Check which score columns exist
    required_score_cols = [cand_A_support_col, cand_B_support_col,
                           cand_A_oppose_col, cand_B_oppose_col]
    existing_score_cols = [c for c in required_score_cols if c in df.columns]

    # Filter to head-to-head polls with political leaning data
    candidate_cols = [f'{c}_percentage' for c in candidates if f'{c}_percentage' in df.columns]
    filter_mask = df[candidate_cols].notna().sum(axis=1) >= 2
    for col in existing_score_cols:
        filter_mask &= df[col].notna()
    head_to_head = df[filter_mask].copy()

    if len(head_to_head) == 0:
        logger.error(f"No head-to-head polls with political leaning data found for: {candidates}")
        return

    logger.info(f"Found {len(head_to_head)} head-to-head polls with political leaning data")

    # Leaning labels derived from config
    pro_positive_label = f"pro-{positive_candidate.lower()}"
    pro_negative_label = f"pro-{negative_candidate.lower()}"

    def categorize_political_leaning(row):
        """
        Categorize poll political leaning based on sentiment scores.

        Returns:
            str: Political leaning category
        """
        positive_indicators = []
        if cand_A_support_col in row.index:
            positive_indicators.append(row[cand_A_support_col] > 0.5)
        if cand_B_oppose_col in row.index:
            positive_indicators.append(row[cand_B_oppose_col] > 0.5)
        if pos_ideology_col in row.index and neg_ideology_col in row.index:
            positive_indicators.append(row[pos_ideology_col] > 0.5 and row[neg_ideology_col] < 0.5)

        negative_indicators = []
        if cand_B_support_col in row.index:
            negative_indicators.append(row[cand_B_support_col] > 0.5)
        if cand_A_oppose_col in row.index:
            negative_indicators.append(row[cand_A_oppose_col] > 0.5)
        if neg_ideology_col in row.index and pos_ideology_col in row.index:
            negative_indicators.append(row[neg_ideology_col] > 0.5 and row[pos_ideology_col] < 0.5)

        if any(positive_indicators) and not any(negative_indicators):
            return pro_positive_label
        elif any(negative_indicators) and not any(positive_indicators):
            return pro_negative_label
        else:
            return 'neutral'

    # Apply political leaning categorization
    head_to_head['political_leaning'] = head_to_head.apply(categorize_political_leaning, axis=1)

    # Log categorization results
    leaning_counts = head_to_head['political_leaning'].value_counts()
    logger.info(f"Political leaning distribution: {dict(leaning_counts)}")

    # Leaning display order and labels
    leaning_order = [pro_positive_label, 'neutral', pro_negative_label]
    leaning_display = [
        f"Pro-{positive_candidate}",
        "Neutral",
        f"Pro-{negative_candidate}",
    ]

    def improved_leaning_scatter(df_candidate, title, ax, color):
        """
        Create sophisticated scatter plot showing political leaning vs vote percentage.

        Args:
            df_candidate: DataFrame with candidate leaning and percentage data
            title: Plot title
            ax: Matplotlib axis object
            color: Scatter point color

        Returns:
            Modified axis object
        """

        # Standardize column names for consistent processing
        candidate_name = df_candidate['candidate'].iloc[0]
        plot_df = df_candidate.rename(columns={
            f'{candidate_name}_percentage': 'percentage'
        }).copy()

        # Remove zero-percentage polls to focus on competitive races
        plot_df_nonzero = plot_df[plot_df['percentage'] > 0].copy()

        if len(plot_df_nonzero) == 0:
            logger.warning(f"No non-zero data for {candidate_name}")
            return ax

        # Map political leaning categories to numeric positions for plotting
        leaning_map = {label: i for i, label in enumerate(leaning_order)}

        # Filter to only existing leaning categories
        available_leanings = [l for l in leaning_order if l in plot_df_nonzero['political_leaning'].values]

        if not available_leanings:
            logger.warning(f"No political leaning data available for {candidate_name}")
            return ax

        # Convert leaning labels to numeric positions
        plot_df_nonzero['leaning_numeric'] = plot_df_nonzero['political_leaning'].map(leaning_map)
        plot_df_nonzero = plot_df_nonzero.dropna(subset=['leaning_numeric'])

        # Add horizontal jitter to scatter points for better visibility
        x_base = plot_df_nonzero['leaning_numeric'].values.astype(float)
        jitter = np.random.normal(loc=0, scale=0.08, size=len(plot_df_nonzero))
        x_jittered = x_base + jitter

        # Background violin plots
        try:
            # Prepare data for violin plot with proper leaning labels
            violin_data = plot_df_nonzero.copy()
            sns.violinplot(x='political_leaning', y='percentage', data=violin_data,
                          order=available_leanings, inner=None, color='lightgray',
                          cut=0, linewidth=0, alpha=0.3, zorder=0, ax=ax)
        except Exception as e:
            logger.warning(f"Could not create violin plot for {candidate_name}: {e}")

        # Main scatter plot with candidate-specific colors
        ax.scatter(
            x_jittered, plot_df_nonzero['percentage'],
            alpha=0.6, s=36, edgecolor='white', linewidth=0.4,
            c=color,
            zorder=3
        )
        grp = plot_df_nonzero.groupby('leaning_numeric')['percentage']
        medians = grp.median()
        means = grp.mean()
        sems = grp.sem().fillna(0)

        x_positions = sorted(plot_df_nonzero['leaning_numeric'].unique())

        if len(x_positions) > 0:
            ax.errorbar(x_positions, means.loc[x_positions], yerr=sems.loc[x_positions],
                       fmt='D', color='black', markersize=7, capsize=5,
                       label='Mean ± SEM', zorder=4)
            for pos in x_positions:
                if pos in medians.index:
                    ax.plot(pos, medians.loc[pos], marker='s', color='darkgreen',
                           markersize=8, zorder=4)

        overall_median = plot_df_nonzero['percentage'].median()
        ax.axhline(overall_median, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Axis labels
        ax.set_xlabel('Political Leaning')
        ax.set_ylabel('Vote Percentage (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Configure x-axis to show leaning categories
        ax.set_xlim(-0.5, len(leaning_order) - 0.5)
        ax.set_xticks(range(len(leaning_order)))
        ax.set_xticklabels(leaning_display)

        # Y-axis covers full percentage range with padding
        ax.set_ylim(-2, 102)

        return ax

    # Setup publication-quality plotting style
    sns.set_style("whitegrid")
    n_candidates = len(candidates)
    fig, axes = plt.subplots(1, n_candidates, figsize=(8 * n_candidates, 7))
    if n_candidates == 1:
        axes = [axes]

    # Prepare candidate-specific datasets
    plot_data = []

    for candidate in candidates:
        candidate_data = head_to_head[['political_leaning', f'{candidate}_percentage']].dropna()

        if len(candidate_data) == 0:
            continue

        candidate_data['candidate'] = candidate
        plot_data.append(candidate_data)

    # Generate multi-panel visualization
    for i, (candidate, cand_data) in enumerate(zip(candidates, plot_data)):
        color = colors.get(candidate, 'tab:gray')
        improved_leaning_scatter(cand_data, f"{candidate} — Political Leaning vs Vote Percentage", axes[i], color)

    # Shared legend explaining statistical markers
    if len(plot_data) >= 2:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='black', linestyle='None',
                   markersize=7, label='Mean ± SEM'),
            Line2D([0], [0], marker='s', color='darkgreen', linestyle='None',
                   markersize=8, label='Median'),
            Line2D([0], [0], color='gray', linestyle='--', label='Overall Median')
        ]

        # Position legend below plots for clean layout
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                  frameon=True, fancybox=True, shadow=True, ncol=3)

    # Save high-resolution plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Plot saved to {output_path}")
    plt.close()

    # Generate detailed statistical summary for reference
    stats_output_path = paths.reports_dir / "political_leaning_statistics.txt"
    with open(stats_output_path, 'w') as f:
        f.write("=== Political Leaning Analysis - Summary Statistics ===\n\n")

        # Overall categorization summary
        f.write("Political Leaning Distribution:\n")
        for leaning_label, count in leaning_counts.items():
            percentage = (count / len(head_to_head)) * 100
            f.write(f"  {leaning_label}: {count} polls ({percentage:.1f}%)\n")
        f.write("\n")

        for i, candidate in enumerate(candidates):
            if i < len(plot_data):
                candidate_data = plot_data[i]
                candidate_nonzero = candidate_data[candidate_data[f'{candidate}_percentage'] > 0]

                f.write(f"{candidate}:\n")
                f.write(f"  Total polls: {len(candidate_data)}\n")
                f.write(f"  Non-zero polls: {len(candidate_nonzero)}\n")

                if len(candidate_nonzero) > 0:
                    # Leaning-specific performance metrics
                    for leaning_label in leaning_order:
                        leaning_data = candidate_nonzero[candidate_nonzero['political_leaning'] == leaning_label]
                        if len(leaning_data) > 0:
                            mean_perf = leaning_data[f'{candidate}_percentage'].mean()
                            median_perf = leaning_data[f'{candidate}_percentage'].median()
                            count = len(leaning_data)
                            f.write(f"  {leaning_label.replace('-', ' ').title()}: {count} polls, Mean: {mean_perf:.1f}%, Median: {median_perf:.1f}%\n")

                    # Overall performance summary
                    overall_mean = candidate_nonzero[f'{candidate}_percentage'].mean()
                    overall_median = candidate_nonzero[f'{candidate}_percentage'].median()
                    f.write(f"  Overall: Mean: {overall_mean:.1f}%, Median: {overall_median:.1f}%\n")

                f.write("\n")

    logger.success(f"Summary statistics saved to {stats_output_path}")

    return plot_data

@app.command()
def bias_relationship_scatter(
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    """
    Generate scatter plots analyzing the relationship between audience bias (x-axis),
    poll outcome (y-axis), and dichotomized bias markers (hue).
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    candidates = ecfg["candidates"]
    column_mapping = ecfg["hypothesis_column_mapping"]
    bias_dir = ecfg["bias_direction"]
    labels = ecfg["label_aliases"]
    positive_candidate = bias_dir["positive"]
    negative_candidate = bias_dir["negative"]
    pos_label = labels["positive_label"]
    neg_label = labels["negative_label"]

    pos_ideology_col = column_mapping.get("positive_ideology", "positive_ideology_score")

    logger.info(f"Starting bias relationship scatter plots for [{election}]...")

    try:
        from bias_analysis.dataset import get_base_dataset

        base_df = get_base_dataset(election=election)

        markers_dir = paths.processed_dir
        candidate_order_df = pd.read_csv(markers_dir / "candidate_order_features.csv") if (markers_dir / "candidate_order_features.csv").exists() else pd.DataFrame()
        appellatives_df = pd.read_csv(markers_dir / "formal_informal_appellatives.csv") if (markers_dir / "formal_informal_appellatives.csv").exists() else pd.DataFrame()
        leaning_df = pd.read_csv(markers_dir / "political_leaning_features.csv") if (markers_dir / "political_leaning_features.csv").exists() else pd.DataFrame()

        unified_df = base_df.copy()

        pos_col = f'{positive_candidate}_position'
        neg_col = f'{negative_candidate}_position'
        pos_pct_col = f'{positive_candidate}_percentage'
        neg_pct_col = f'{negative_candidate}_percentage'

        if not candidate_order_df.empty:
            candidate_order_df['poll_id'] = candidate_order_df['poll_id'].astype(str)
            merge_cols = ['poll_id'] + [c for c in [pos_col, neg_col, pos_pct_col, neg_pct_col] if c in candidate_order_df.columns]
            unified_df = unified_df.merge(candidate_order_df[merge_cols],
                                          left_on='tweet_id', right_on='poll_id', how='left')
            if neg_col in unified_df.columns and pos_col in unified_df.columns:
                unified_df['candidate_order'] = (unified_df[neg_col] - unified_df[pos_col]) * 0.25

        if not appellatives_df.empty:
            appellatives_df['poll_id'] = appellatives_df['poll_id'].astype(str)
            app_merge_cols = ['poll_id']
            for c in candidates:
                score_col = f'{c}_formality_score'
                if score_col in appellatives_df.columns:
                    app_merge_cols.append(score_col)
                elif f'{c}_label' in appellatives_df.columns:
                    app_merge_cols.append(f'{c}_label')
            unified_df = unified_df.merge(appellatives_df[app_merge_cols],
                                          left_on='tweet_id', right_on='poll_id', how='left')
            for c in candidates:
                score_col = f'{c}_formality_score'
                if score_col in unified_df.columns:
                    unified_df[f'{c.lower()}_formal_appellative'] = (
                        (unified_df[score_col] - 2.5) / 2.5
                    )
                elif f'{c}_label' in unified_df.columns:
                    label_map = {'formal': 1.0, 'informal': -1.0, 'neutral': 0.0}
                    unified_df[f'{c.lower()}_formal_appellative'] = unified_df[f'{c}_label'].map(label_map)

        leaning_score_cols = list(column_mapping.values())

        if not leaning_df.empty:
            leaning_df['poll_id'] = leaning_df['poll_id'].astype(str)
            leaning_merge_cols = ['poll_id'] + [c for c in leaning_score_cols if c in leaning_df.columns]
            unified_df = unified_df.merge(leaning_df[leaning_merge_cols],
                                          left_on='tweet_id', right_on='poll_id', how='left')
            if pos_ideology_col in unified_df.columns:
                unified_df['text_partisan_score'] = unified_df[pos_ideology_col]

        if pos_pct_col in unified_df.columns:
            unified_df[f'{positive_candidate.lower()}_share'] = unified_df[pos_pct_col] / 100.0
        else:
            unified_df[f'{positive_candidate.lower()}_share'] = np.nan
        if neg_pct_col in unified_df.columns:
            unified_df[f'{negative_candidate.lower()}_share'] = unified_df[neg_pct_col] / 100.0
        else:
            unified_df[f'{negative_candidate.lower()}_share'] = np.nan

    except Exception as e:
        logger.error(f"Failed to load unified dataset: {e}")
        return

    if unified_df.empty:
        logger.error("Unified dataset is empty!")
        return

    # Create audience_bias and poll_outcome measures
    def normalize_audience_bias(partisanship_score):
        if pd.notna(partisanship_score):
            return np.tanh(partisanship_score)
        return np.nan

    unified_df['audience_bias'] = unified_df['audience_mean_partisanship'].apply(normalize_audience_bias)

    pos_share_col = f'{positive_candidate.lower()}_share'
    neg_share_col = f'{negative_candidate.lower()}_share'

    def compute_poll_outcome_bias(row):
        pos_share = row.get(pos_share_col)
        neg_share = row.get(neg_share_col)
        if pd.notna(pos_share) and pd.notna(neg_share):
            return np.clip(pos_share - neg_share, -1.0, 1.0)
        return np.nan

    unified_df['poll_outcome'] = unified_df.apply(compute_poll_outcome_bias, axis=1)

    # Formality bias
    pos_formal_col = f'{positive_candidate.lower()}_formal_appellative'
    neg_formal_col = f'{negative_candidate.lower()}_formal_appellative'

    def compute_formality_bias(row):
        pos_f = row.get(pos_formal_col, np.nan)
        neg_f = row.get(neg_formal_col, np.nan)
        if pd.isna(pos_f) and pd.isna(neg_f):
            return np.nan
        pos_v = float(pos_f) if pd.notna(pos_f) else 0.0
        neg_v = float(neg_f) if pd.notna(neg_f) else 0.0
        return float(np.clip(pos_v - neg_v, -1.0, 1.0))

    unified_df['formality_bias'] = unified_df.apply(compute_formality_bias, axis=1)

    markers = {
        'candidate_order': 'Candidate Order Bias',
        'formality_bias': 'Formality Bias',
        'text_partisan_score': 'Political Leaning (Text)'
    }

    sns.set_style("whitegrid")

    for marker_col, marker_name in markers.items():
        if marker_col not in unified_df.columns:
            logger.warning(f"Marker {marker_col} missing, skipping...")
            continue

        plot_df = unified_df[['audience_bias', 'poll_outcome', marker_col]].dropna().copy()

        if len(plot_df) < 10:
            logger.warning(f"Not enough data for {marker_col}")
            continue

        median_val = plot_df[marker_col].median()

        # Categorize around median
        epsilon = 1e-5
        def categorize_marker(val):
            if val > median_val + epsilon:
                return 'High'
            elif val < median_val - epsilon:
                return 'Low'
            else:
                return 'Neutral'

        plot_df['marker_categorized'] = plot_df[marker_col].apply(categorize_marker)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=plot_df,
            x='audience_bias',
            y='poll_outcome',
            hue='marker_categorized',
            palette={'High': 'tab:red', 'Neutral': 'tab:gray', 'Low': 'tab:blue'},
            hue_order=['High', 'Neutral', 'Low'],
            alpha=0.6,
            s=50,
            edgecolor='white'
        )

        # Add zero lines
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

        plt.title(f'Relationship Between Audience Bias and Poll Outcome\nHue: {marker_name} (Categorized around Median: {median_val:.3f})')
        plt.xlabel(f'Audience Bias (-1=Left/{neg_label}, +1=Right/{pos_label})')
        plt.ylabel(f'Poll Outcome Bias (-1={negative_candidate} win, +1={positive_candidate} win)')
        plt.legend(title=f'{marker_name}\n(High=Right/{positive_candidate}-leaning, Low=Left/{negative_candidate}-leaning)')

        output_path = paths.figures_dir / f"scatter_{marker_col}_bias.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.success(f"Saved scatter plot to {output_path}")

@app.command()
def generate_adjusted_dashboard(
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    """
    Generate an interactive Plotly time-series dashboard showing rolling
    SCWG-adjusted Twitter poll estimates, raw Twitter means, PredictIt market
    prices, and actual results on a single unified chart.

    Saves output as an interactive HTML file in the election's reports directory.
    """
    import plotly.graph_objects as go

    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    candidates = ecfg["candidates"]
    colors = ecfg["candidate_colors"]
    bias_dir = ecfg["bias_direction"]
    positive_candidate = bias_dir["positive"]
    negative_candidate = bias_dir["negative"]
    milestones = ecfg.get("milestones", [])
    election_date_str = ecfg.get("election_date", None)

    from bias_analysis.dataset import get_base_dataset
    from bias_analysis.modeling.train import (
        load_and_merge_features,
        fit_glm,
        fit_glm_baseline,
        build_poststrat_frame,
        poststratify,
    )

    logger.info(f"Loading data and fitting global GLM for SCWG dashboard [{election}] …")
    base_df = get_base_dataset(election=election)
    if base_df.empty:
        logger.error("Base dataset is empty — aborting.")
        return

    actual_pos_share = ecfg["mrp"]["actual_results"][positive_candidate]
    actual_neg_share = ecfg["mrp"]["actual_results"][negative_candidate]

    df = load_and_merge_features(base_df, election=election)
    analysis_df = df.dropna(subset=["positive_share"]).copy()
    logger.info(f"Analysis-ready polls: {len(analysis_df)}")

    if len(analysis_df) < 10:
        logger.error("Too few polls — aborting.")
        return

    # Single global GLM fit
    glm_result = fit_glm(analysis_df)
    baseline_glm_result = fit_glm_baseline(analysis_df)
    logger.info("Global GLMs fitted successfully.")

    # Parse dates
    analysis_df["date"] = pd.to_datetime(
        analysis_df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
    )
    analysis_df = analysis_df.dropna(subset=["date"])
    analysis_df["date_only"] = analysis_df["date"].dt.date

    # Determine date range
    min_date = analysis_df["date_only"].min()
    max_date = analysis_df["date_only"].max()

    if election_date_str:
        election_day = datetime.strptime(election_date_str, "%Y-%m-%d").date()
        end_date = min(max_date, election_day)
    else:
        end_date = max_date

    logger.info(f"Date range: {min_date} → {end_date}")

    # Rolling window computation
    window_days = timedelta(days=7)
    results = []

    current = pd.Timestamp(min_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")

    while current <= end_ts:
        window_start = current - window_days
        window_end = current
        mask = (analysis_df["date"] > window_start) & (analysis_df["date"] <= window_end)
        window_df = analysis_df.loc[mask]

        if len(window_df) >= 3:
            raw_share = window_df["positive_share"].mean()

            # Window-specific post-stratification
            ps_frame = build_poststrat_frame(window_df)
            scwg_share = poststratify(glm_result, ps_frame)
            baseline_scwg_share = poststratify(baseline_glm_result, ps_frame)

            results.append({
                "date": current.date(),
                f"raw_{positive_candidate.lower()}": raw_share,
                f"raw_{negative_candidate.lower()}": 1.0 - raw_share,
                f"scwg_{positive_candidate.lower()}": scwg_share,
                f"scwg_{negative_candidate.lower()}": 1.0 - scwg_share,
                f"baseline_{positive_candidate.lower()}": baseline_scwg_share,
                f"baseline_{negative_candidate.lower()}": 1.0 - baseline_scwg_share,
                "n_polls": len(window_df),
            })

        current += timedelta(days=1)

    if not results:
        logger.error("No rolling windows had enough polls, aborting.")
        return

    rdf = pd.DataFrame(results)
    rdf["date"] = pd.to_datetime(rdf["date"])

    logger.info(f"Computed {len(rdf)} daily data points for the dashboard.")

    # --- Load PredictIt prediction market data --------------------------------
    pit = load_predictit_data(election)
    predictit_df = None

    if not pit.empty:
        try:
            # Filter to main candidates using substring matching
            # (contract names are full names like "Donald Trump", our config uses "Trump")
            pit["_name"] = pit["Contract Name"].str.strip()
            # We use "Close Share Price" which was cleaned in load_predictit_data
            pos_pit = pit[pit["_name"].str.contains(positive_candidate, case=False, na=False)].copy()
            neg_pit = pit[pit["_name"].str.contains(negative_candidate, case=False, na=False)].copy()

            if not pos_pit.empty and not neg_pit.empty:
                pos_pit = pos_pit[["date", "Close Share Price"]].rename(columns={"Close Share Price": f"market_{positive_candidate.lower()}"})
                neg_pit = neg_pit[["date", "Close Share Price"]].rename(columns={"Close Share Price": f"market_{negative_candidate.lower()}"})

                predictit_df = pos_pit.merge(neg_pit, on="date", how="outer").sort_values("date")

                # Filter to our analysis date range
                predictit_df = predictit_df[
                    (predictit_df["date"] >= pd.Timestamp(min(rdf["date"])))
                    & (predictit_df["date"] < pd.Timestamp(end_date)) # Strictly less than end_date to drop the final day, since results start to come out on election day and market closes after.
                ]

                logger.success(f"Processed {len(predictit_df)} PredictIt daily records for {positive_candidate} vs {negative_candidate}.")
            else:
                logger.warning("Could not find both candidates in PredictIt data.")
        except Exception as e:
            logger.warning(f"Failed to process PredictIt data: {e}")
    else:
        logger.warning(f"No PredictIt data available for {election}, skipping market traces.")

    # --- Build colour lookup --------------------------------------------------
    pos_color = colors.get(positive_candidate, "#DC3545")
    neg_color = colors.get(negative_candidate, "#0D6EFD")
    pos_hex = _mpl_color_to_hex(pos_color)
    neg_hex = _mpl_color_to_hex(neg_color)

    # Build Plotly figure
    fig = go.Figure()

    pos_scwg_col = f"scwg_{positive_candidate.lower()}"
    neg_scwg_col = f"scwg_{negative_candidate.lower()}"
    pos_raw_col = f"raw_{positive_candidate.lower()}"
    neg_raw_col = f"raw_{negative_candidate.lower()}"

    # SCWG positive candidate (solid)
    fig.add_trace(go.Scatter(
        x=rdf["date"], y=rdf[pos_scwg_col] * 100,
        mode="lines", name=f"SCWG Estimate for {positive_candidate}",
        line=dict(color=pos_hex, width=3),
        legendgroup="scwg",
        hovertemplate=f"<b>{positive_candidate} SCWG</b>: %{{y:.1f}}%<extra></extra>",
    ))

    # SCWG negative candidate (solid)
    fig.add_trace(go.Scatter(
        x=rdf["date"], y=rdf[neg_scwg_col] * 100,
        mode="lines", name=f"SCWG Estimate for {negative_candidate}",
        line=dict(color=neg_hex, width=3),
        legendgroup="scwg",
        hovertemplate=f"<b>{negative_candidate} SCWG</b>: %{{y:.1f}}%<extra></extra>",
    ))

    # Raw positive candidate (dashed)
    fig.add_trace(go.Scatter(
        x=rdf["date"], y=rdf[pos_raw_col] * 100,
        mode="lines", name=f"Raw Twitter Mean for {positive_candidate}",
        line=dict(color=pos_hex, width=1.5, dash="dot"),
        opacity=0.8,
        legendgroup="raw",
        hovertemplate=f"<b>{positive_candidate} Raw</b>: %{{y:.1f}}%<extra></extra>",
    ))

    # Raw negative candidate (dashed)
    fig.add_trace(go.Scatter(
        x=rdf["date"], y=rdf[neg_raw_col] * 100,
        mode="lines", name=f"Raw Twitter Mean for {negative_candidate}",
        line=dict(color=neg_hex, width=1.5, dash="dot"),
        opacity=0.8,
        legendgroup="raw",
        hovertemplate=f"<b>{negative_candidate} Raw</b>: %{{y:.1f}}%<extra></extra>",
    ))

    pos_base_col = f"baseline_{positive_candidate.lower()}"
    neg_base_col = f"baseline_{negative_candidate.lower()}"

    # Baseline positive candidate
    fig.add_trace(go.Scatter(
        x=rdf["date"], y=rdf[pos_base_col] * 100,
        mode="lines", name=f"Baseline SCWG for {positive_candidate}",
        line=dict(color=pos_hex, width=2, dash="dashdot"),
        legendgroup="baseline",
        hovertemplate=f"<b>{positive_candidate} Baseline</b>: %{{y:.1f}}%<extra></extra>",
        visible=False,
    ))

    # Baseline negative candidate
    fig.add_trace(go.Scatter(
        x=rdf["date"], y=rdf[neg_base_col] * 100,
        mode="lines", name=f"Baseline SCWG for {negative_candidate}",
        line=dict(color=neg_hex, width=2, dash="dashdot"),
        legendgroup="baseline",
        hovertemplate=f"<b>{negative_candidate} Baseline</b>: %{{y:.1f}}%<extra></extra>",
        visible=False,
    ))

    # PredictIt market time-series (dashdot)
    if predictit_df is not None and not predictit_df.empty:
        pos_mkt_col = f"market_{positive_candidate.lower()}"
        neg_mkt_col = f"market_{negative_candidate.lower()}"

        fig.add_trace(go.Scatter(
            x=predictit_df["date"], y=predictit_df[pos_mkt_col] * 100,
            mode="lines", name=f"PredictIt Close Price for {positive_candidate}",
            line=dict(color=pos_hex, width=1.5, dash="dash"),
            opacity=0.5,
            legendgroup="market",
            visible="legendonly",
            hovertemplate=f"<b>{positive_candidate} Market</b>: %{{y:.1f}}%<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=predictit_df["date"], y=predictit_df[neg_mkt_col] * 100,
            mode="lines", name=f"PredictIt Close Price for {negative_candidate}",
            line=dict(color=neg_hex, width=1.5, dash="dash"),
            opacity=0.5,
            legendgroup="market",
            visible="legendonly",
            hovertemplate=f"<b>{negative_candidate} Market</b>: %{{y:.1f}}%<extra></extra>",
        ))

    # Poll Count (invisible, for hover info)
    fig.add_trace(go.Scatter(
        x=rdf["date"], y=[50] * len(rdf),
        mode="lines", name="Polls in window",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False, hoverinfo="name+text",
        customdata=rdf["n_polls"],
        hovertemplate="<b>Polls in window</b>: %{customdata}<extra></extra>",
    ))

    # Actual result lines (grey)
    x_bounds = [rdf["date"].min(), rdf["date"].max()]

    fig.add_trace(go.Scatter(
        x=x_bounds, y=[actual_pos_share * 100, actual_pos_share * 100],
        mode="lines", name="Actual Results", legendgroup="actuals",
        line=dict(color="#6C757D", width=2), opacity=0.7,
        hovertemplate=f"<b>Actual - {positive_candidate}</b>: %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_bounds, y=[actual_neg_share * 100, actual_neg_share * 100],
        mode="lines", name="Actual Results", legendgroup="actuals", showlegend=False,
        line=dict(color="#ADB5BD", width=2), opacity=0.7,
        hovertemplate=f"<b>Actual - {negative_candidate}</b>: %{{y:.1f}}%<extra></extra>",
    ))

    # Campaign milestones from election config
    for ms_date_str, ms_text in milestones:
        ms_date = datetime.strptime(ms_date_str, "%Y-%m-%d")
        ms_timestamp = ms_date.timestamp() * 1000

        fig.add_vline(
            x=ms_timestamp,
            line_width=1.5,
            line_dash="dash",
            line_color="#6F42C1",
            opacity=0.8
        )

        fig.add_trace(go.Scatter(
            x=[ms_date], y=[50],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=0.1),
            showlegend=False,
            hovertemplate=f"<b>Event</b> ({ms_date.strftime('%Y-%m-%d')}):<br>{ms_text}<extra></extra>",
        ))

    display_name = ecfg.get("display_name", election)

    fig.update_layout(
        title=dict(
            text=f"SCWG-Adjusted Twitter Polls vs Benchmarks ({display_name})",
            font=dict(size=20),
        ),
        xaxis_title="Date",
        yaxis_title="Vote Share (%)",
        yaxis=dict(range=[0, 100], dtick=10),
        legend=dict(
            orientation="h", yanchor="top", y=-0.1,
            xanchor="center", x=0.5,
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.5,
                y=-0.25,
                xanchor="center",
                yanchor="top",
                buttons=list([
                    dict(label="SCWG vs Raw",
                         method="restyle",
                         args=[{"visible": [True, True, True, True, False, False]}, [0, 1, 2, 3, 4, 5]]),
                    dict(label="SCWG vs Baseline",
                         method="restyle",
                         args=[{"visible": [True, True, False, False, True, True]}, [0, 1, 2, 3, 4, 5]]),
                ]),
            )
        ],
        hovermode="x unified",
        template="plotly_white",
        height=650,
        margin=dict(b=150, t=100),
    )

    # Save
    output_path = paths.reports_dir / "scwg_bias_dashboard.html"
    fig.write_html(str(output_path), include_plotlyjs=True)
    logger.success(f"Interactive dashboard saved to {output_path}")

    # Compute MAE vs actual result
    mae_actual = (rdf[pos_scwg_col] - actual_pos_share).abs().mean() * 100
    logger.info(f"SCWG Estimate MAE vs Actual Result (across all days): {mae_actual:.2f} pp")

    # Compute additional metrics if PredictIt Market is available
    if predictit_df is not None and not predictit_df.empty:
        mkt_col = f"market_{positive_candidate.lower()}"
        
        # 1. Market MAE vs Actual
        mae_mkt_actual = (predictit_df[mkt_col] - actual_pos_share).abs().mean() * 100
        logger.info(f"PredictIt Market MAE vs Actual Result (across all market days): {mae_mkt_actual:.2f} pp")

        # 2. Divergence between SCWG and Market
        merged = rdf.merge(predictit_df[["date", mkt_col]], on="date", how="inner")
        if not merged.empty:
            divergence = (merged[pos_scwg_col] - merged[mkt_col]).abs().mean() * 100
            logger.info(f"Mean Divergence (MAE) between SCWG and PredictIt: {divergence:.2f} pp")

    return rdf


def _mpl_color_to_hex(color_str: str) -> str:
    """
    Convert a matplotlib named colour (e.g. 'tab:red') to a hex string
    that Plotly understands. Falls back to the input if conversion fails.
    """
    try:
        import matplotlib.colors as mcolors
        rgba = mcolors.to_rgba(color_str)
        return "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
    except Exception:
        return color_str


@app.command()
def run_all(
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    """
    Run all plotting functions to generate complete set of visualizations.
    """
    candidate_order(election=election)
    appellatives(election=election)
    leaning(election=election)
    bias_relationship_scatter(election=election)
    generate_adjusted_dashboard(election=election)

if __name__ == "__main__":
    app()
