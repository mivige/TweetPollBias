"""
Visualization module for Twitter poll bias analysis.

This module creates publication-quality visualizations to analyze position bias
in Twitter polls, specifically examining how candidate order affects voting outcomes
in the 2020 US presidential election polls.
"""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from bias_analysis.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

@app.command()
def candidate_order():
    """
    Generate advanced scatter plots analyzing candidate position bias in Twitter polls.
    
    Creates dual-panel visualization showing how poll position correlates with vote percentage.
    Saves both the plot and detailed statistics to files.
    """
    input_path = PROCESSED_DATA_DIR / "candidate_order_features.csv"
    output_path = FIGURES_DIR / "candidate_order_plot.png"

    # Load processed candidate order features
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} poll records")

    # Filter to Trump vs Biden head-to-head polls for focused analysis
    trump_biden_polls = df[(df['Trump_position'].notna()) & (df['Biden_position'].notna())].copy()
    
    if len(trump_biden_polls) == 0:
        logger.error("No Trump vs Biden polls found!")
        return
        
    logger.info(f"Found {len(trump_biden_polls)} Trump vs Biden polls for analysis")

    def improved_scatter(df_candidate, title, ax):
        """
        Create sophisticated scatter plot with multiple statistical layers.
        
        Args:
            df_candidate: DataFrame with candidate position and percentage data
            title: Plot title
            ax: Matplotlib axis object
            
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
            c='tab:red' if candidate_name == 'Trump' else 'tab:blue',
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prepare candidate-specific datasets
    plot_data = []
    
    for candidate in ['Trump', 'Biden']:
        candidate_data = trump_biden_polls[[f'{candidate}_position', f'{candidate}_percentage']].dropna()
        
        if len(candidate_data) == 0:
            continue
            
        candidate_data['candidate'] = candidate
        plot_data.append(candidate_data)
    
    # Generate dual-panel visualization
    if len(plot_data) >= 2:
        # Left panel: Trump position bias analysis
        trump_data = plot_data[0]
        improved_scatter(trump_data, "Trump — Position vs Vote Percentage", ax1)
        
        # Right panel: Biden position bias analysis
        biden_data = plot_data[1]
        improved_scatter(biden_data, "Biden — Position vs Vote Percentage", ax2)
        
        # Shared legend explaining statistical markers
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
    stats_output_path = REPORTS_DIR / "candidate_order_statistics.txt"
    with open(stats_output_path, 'w') as f:
        f.write("=== Candidate Order Analysis - Summary Statistics ===\n\n")
        
        for i, candidate in enumerate(['Trump', 'Biden']):
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
def appellatives():
    """
    Generate advanced scatter plots analyzing formality bias in Twitter poll appellatives.
    
    Creates dual-panel visualization showing how appellative formality correlates with vote percentage.
    """
    input_path = PROCESSED_DATA_DIR / "formal_informal_appellatives.csv"
    output_path = FIGURES_DIR / "formal_informal_appellatives_plot.png"

    # Load processed appellative features
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} poll records")

    # Filter to Trump vs Biden head-to-head polls with formality data
    trump_biden_polls = df[
        (df['Trump_label'].notna()) & (df['Biden_label'].notna()) &
        (df['Trump_percentage'].notna()) & (df['Biden_percentage'].notna())
    ].copy()
    
    if len(trump_biden_polls) == 0:
        logger.error("No Trump vs Biden polls with formality data found!")
        return
        
    logger.info(f"Found {len(trump_biden_polls)} Trump vs Biden polls with appellative data")

    def improved_formality_scatter(df_candidate, title, ax):
        """
        Create sophisticated scatter plot showing formality vs vote percentage.
        
        Args:
            df_candidate: DataFrame with candidate formality and percentage data
            title: Plot title
            ax: Matplotlib axis object
            
        Returns:
            Modified axis object
        """
        
        # Standardize column names for consistent processing
        candidate_name = df_candidate['candidate'].iloc[0]
        plot_df = df_candidate.rename(columns={
            f'{candidate_name}_label': 'formality',
            f'{candidate_name}_percentage': 'percentage'
        }).copy()
        
        # Remove zero-percentage polls to focus on competitive races
        plot_df_nonzero = plot_df[plot_df['percentage'] > 0].copy()
        
        if len(plot_df_nonzero) == 0:
            logger.warning(f"No non-zero data for {candidate_name}")
            return ax
            
        # Map formality categories to numeric positions for plotting
        formality_order = ['informal', 'neutral', 'formal']
        formality_map = {label: i for i, label in enumerate(formality_order)}
        
        # Filter to only existing formality categories
        available_formalities = [f for f in formality_order if f in plot_df_nonzero['formality'].values]
        
        if not available_formalities:
            logger.warning(f"No formality data available for {candidate_name}")
            return ax
        
        # Convert formality labels to numeric positions
        plot_df_nonzero['formality_numeric'] = plot_df_nonzero['formality'].map(formality_map)
        plot_df_nonzero = plot_df_nonzero.dropna(subset=['formality_numeric'])
        
        # Add horizontal jitter to scatter points for better visibility
        x_base = plot_df_nonzero['formality_numeric'].values.astype(float)
        jitter = np.random.normal(loc=0, scale=0.08, size=len(plot_df_nonzero))
        x_jittered = x_base + jitter
        
        # Background violin plots
        try:
            # Prepare data for violin plot with proper formality labels
            violin_data = plot_df_nonzero.copy()
            sns.violinplot(x='formality', y='percentage', data=violin_data,
                          order=available_formalities, inner=None, color='lightgray', 
                          cut=0, linewidth=0, alpha=0.3, zorder=0, ax=ax)
        except Exception as e:
            logger.warning(f"Could not create violin plot for {candidate_name}: {e}")
        
        # Main scatter plot with candidate-specific colors
        ax.scatter(
            x_jittered, plot_df_nonzero['percentage'],
            alpha=0.6, s=36, edgecolor='white', linewidth=0.4,
            c='tab:red' if candidate_name == 'Trump' else 'tab:blue',
            zorder=3
        )
        
        # Calculate formality-wise statistics for overlays
        grp = plot_df_nonzero.groupby('formality_numeric')['percentage']
        medians = grp.median()
        means = grp.mean()
        sems = grp.sem().fillna(0)  # Standard error of mean
        
        x_positions = sorted(plot_df_nonzero['formality_numeric'].unique())
        
        # Overlay mean values with error bars showing uncertainty
        if len(x_positions) > 0:
            ax.errorbar(x_positions, means.loc[x_positions], yerr=sems.loc[x_positions],
                       fmt='D', color='black', markersize=7, capsize=5, 
                       label='Mean ± SEM', zorder=4)
            
            # Overlay median markers (robust central tendency)
            for pos in x_positions:
                if pos in medians.index:
                    ax.plot(pos, medians.loc[pos], marker='s', color='darkgreen', 
                           markersize=8, zorder=4)
        
        # Reference line: overall median
        overall_median = plot_df_nonzero['percentage'].median()
        ax.axhline(overall_median, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        # Axis labels
        ax.set_xlabel('Appellative Formality')
        ax.set_ylabel('Vote Percentage (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Configure x-axis to show formality categories
        ax.set_xlim(-0.5, len(formality_order) - 0.5)
        ax.set_xticks(range(len(formality_order)))
        ax.set_xticklabels([label.capitalize() for label in formality_order])
        
        # Y-axis covers full percentage range with padding
        ax.set_ylim(-2, 102)
        
        return ax

    # Setup publication-quality plotting style
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prepare candidate-specific datasets
    plot_data = []
    
    for candidate in ['Trump', 'Biden']:
        candidate_data = trump_biden_polls[[f'{candidate}_label', f'{candidate}_percentage']].dropna()
        
        if len(candidate_data) == 0:
            continue
            
        candidate_data['candidate'] = candidate
        plot_data.append(candidate_data)
    
    # Generate dual-panel visualization
    if len(plot_data) >= 2:
        # Left panel: Trump formality bias analysis
        trump_data = plot_data[0]
        improved_formality_scatter(trump_data, "Trump — Appellative Formality vs Vote Percentage", ax1)
        
        # Right panel: Biden formality bias analysis
        biden_data = plot_data[1]
        improved_formality_scatter(biden_data, "Biden — Appellative Formality vs Vote Percentage", ax2)
        
        # Shared legend explaining statistical markers
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
    stats_output_path = REPORTS_DIR / "formal_informal_appellatives_statistics.txt"
    with open(stats_output_path, 'w') as f:
        f.write("=== Formal vs Informal Appellatives Analysis - Summary Statistics ===\n\n")
        
        for i, candidate in enumerate(['Trump', 'Biden']):
            if i < len(plot_data):
                candidate_data = plot_data[i]
                candidate_nonzero = candidate_data[candidate_data[f'{candidate}_percentage'] > 0]
                
                f.write(f"{candidate}:\n")
                f.write(f"  Total polls: {len(candidate_data)}\n")
                f.write(f"  Non-zero polls: {len(candidate_nonzero)}\n")
                
                if len(candidate_nonzero) > 0:
                    # Formality-specific performance metrics
                    for formality in ['informal', 'neutral', 'formal']:
                        formality_data = candidate_nonzero[candidate_nonzero[f'{candidate}_label'] == formality]
                        if len(formality_data) > 0:
                            mean_perf = formality_data[f'{candidate}_percentage'].mean()
                            median_perf = formality_data[f'{candidate}_percentage'].median()
                            count = len(formality_data)
                            f.write(f"  {formality.capitalize()}: {count} polls, Mean: {mean_perf:.1f}%, Median: {median_perf:.1f}%\n")
                    
                    # Overall performance summary
                    overall_mean = candidate_nonzero[f'{candidate}_percentage'].mean()
                    overall_median = candidate_nonzero[f'{candidate}_percentage'].median()
                    f.write(f"  Overall: Mean: {overall_mean:.1f}%, Median: {overall_median:.1f}%\n")
                
                f.write("\n")
    
    logger.success(f"Summary statistics saved to {stats_output_path}")

    return plot_data

@app.command()
def leaning():
    """
    Generate advanced scatter plots analyzing political leaning bias in Twitter polls.
    
    Creates dual-panel visualization showing how political leaning correlates with vote percentage.
    """
    input_path = PROCESSED_DATA_DIR / "political_leaning_features.csv"
    output_path = FIGURES_DIR / "political_leaning_plot.png"

    # Load processed political leaning features
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} poll records")

    # Filter to Trump vs Biden head-to-head polls with political leaning data
    trump_biden_polls = df[
        (df['Trump_percentage'].notna()) & (df['Biden_percentage'].notna()) &
        (df['trump_support_score'].notna()) & (df['biden_support_score'].notna()) &
        (df['anti_trump_score'].notna()) & (df['anti_biden_score'].notna())
    ].copy()
    
    if len(trump_biden_polls) == 0:
        logger.error("No Trump vs Biden polls with political leaning data found!")
        return
        
    logger.info(f"Found {len(trump_biden_polls)} Trump vs Biden polls with political leaning data")

    def categorize_political_leaning(row):
        """
        Categorize poll political leaning based on sentiment scores.
        
        Args:
            row: DataFrame row with political scores
            
        Returns:
            str: Political leaning category ('pro-trump', 'neutral', 'pro-biden')
        """
        trump_indicators = [
            row['trump_support_score'] > 0.5,
            row['anti_biden_score'] > 0.5,
            (row['conservative_score'] > 0.5 and row['liberal_score'] < 0.5)
        ]
        
        biden_indicators = [
            row['biden_support_score'] > 0.5, 
            row['anti_trump_score'] > 0.5,
            (row['liberal_score'] > 0.5 and row['conservative_score'] < 0.5)
        ]
        if any(trump_indicators) and not any(biden_indicators):
            return 'pro-trump'
        elif any(biden_indicators) and not any(trump_indicators):
            return 'pro-biden'
        else:
            return 'neutral'

    # Apply political leaning categorization
    trump_biden_polls['political_leaning'] = trump_biden_polls.apply(categorize_political_leaning, axis=1)
    
    # Log categorization results
    leaning_counts = trump_biden_polls['political_leaning'].value_counts()
    logger.info(f"Political leaning distribution: {dict(leaning_counts)}")

    def improved_leaning_scatter(df_candidate, title, ax):
        """
        Create sophisticated scatter plot showing political leaning vs vote percentage.
        
        Args:
            df_candidate: DataFrame with candidate leaning and percentage data
            title: Plot title
            ax: Matplotlib axis object
            
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
        leaning_order = ['pro-trump', 'neutral', 'pro-biden']
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
            c='tab:red' if candidate_name == 'Trump' else 'tab:blue',
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
        ax.set_xticklabels(['Pro-Trump', 'Neutral', 'Pro-Biden'])
        
        # Y-axis covers full percentage range with padding
        ax.set_ylim(-2, 102)
        
        return ax

    # Setup publication-quality plotting style
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prepare candidate-specific datasets
    plot_data = []
    
    for candidate in ['Trump', 'Biden']:
        candidate_data = trump_biden_polls[['political_leaning', f'{candidate}_percentage']].dropna()
        
        if len(candidate_data) == 0:
            continue
            
        candidate_data['candidate'] = candidate
        plot_data.append(candidate_data)
    
    # Generate dual-panel visualization
    if len(plot_data) >= 2:
        # Left panel: Trump political leaning bias analysis
        trump_data = plot_data[0]
        improved_leaning_scatter(trump_data, "Trump — Political Leaning vs Vote Percentage", ax1)
        
        # Right panel: Biden political leaning bias analysis
        biden_data = plot_data[1]
        improved_leaning_scatter(biden_data, "Biden — Political Leaning vs Vote Percentage", ax2)
        
        # Shared legend explaining statistical markers
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
    stats_output_path = REPORTS_DIR / "political_leaning_statistics.txt"
    with open(stats_output_path, 'w') as f:
        f.write("=== Political Leaning Analysis - Summary Statistics ===\n\n")
        
        # Overall categorization summary
        f.write("Political Leaning Distribution:\n")
        for leaning, count in leaning_counts.items():
            percentage = (count / len(trump_biden_polls)) * 100
            f.write(f"  {leaning}: {count} polls ({percentage:.1f}%)\n")
        f.write("\n")
        
        for i, candidate in enumerate(['Trump', 'Biden']):
            if i < len(plot_data):
                candidate_data = plot_data[i]
                candidate_nonzero = candidate_data[candidate_data[f'{candidate}_percentage'] > 0]
                
                f.write(f"{candidate}:\n")
                f.write(f"  Total polls: {len(candidate_data)}\n")
                f.write(f"  Non-zero polls: {len(candidate_nonzero)}\n")
                
                if len(candidate_nonzero) > 0:
                    # Leaning-specific performance metrics
                    for leaning in ['pro-trump', 'neutral', 'pro-biden']:
                        leaning_data = candidate_nonzero[candidate_nonzero['political_leaning'] == leaning]
                        if len(leaning_data) > 0:
                            mean_perf = leaning_data[f'{candidate}_percentage'].mean()
                            median_perf = leaning_data[f'{candidate}_percentage'].median()
                            count = len(leaning_data)
                            f.write(f"  {leaning.replace('-', ' ').title()}: {count} polls, Mean: {mean_perf:.1f}%, Median: {median_perf:.1f}%\n")
                    
                    # Overall performance summary
                    overall_mean = candidate_nonzero[f'{candidate}_percentage'].mean()
                    overall_median = candidate_nonzero[f'{candidate}_percentage'].median()
                    f.write(f"  Overall: Mean: {overall_mean:.1f}%, Median: {overall_median:.1f}%\n")
                
                f.write("\n")
    
    logger.success(f"Summary statistics saved to {stats_output_path}")

    return plot_data

@app.command()
def bias_relationship_scatter():
    """
    Generate scatter plots analyzing the relationship between audience bias (x-axis),
    poll outcome (y-axis), and dichotomized bias markers (hue).
    """
    logger.info("Starting bias relationship scatter plots...")
    
    try:
        from bias_analysis.dataset import get_base_dataset
        
        base_df = get_base_dataset()
        
        markers_dir = PROCESSED_DATA_DIR
        candidate_order_df = pd.read_csv(markers_dir / "candidate_order_features.csv") if (markers_dir / "candidate_order_features.csv").exists() else pd.DataFrame()
        appellatives_df = pd.read_csv(markers_dir / "formal_informal_appellatives.csv") if (markers_dir / "formal_informal_appellatives.csv").exists() else pd.DataFrame()
        leaning_df = pd.read_csv(markers_dir / "political_leaning_features.csv") if (markers_dir / "political_leaning_features.csv").exists() else pd.DataFrame()
        
        unified_df = base_df.copy()
        
        if not candidate_order_df.empty:
            candidate_order_df['poll_id'] = candidate_order_df['poll_id'].astype(str)
            unified_df = unified_df.merge(candidate_order_df[['poll_id', 'Trump_position', 'Biden_position', 'Trump_percentage', 'Biden_percentage']], 
                                          left_on='tweet_id', right_on='poll_id', how='left')
            unified_df['candidate_order'] = (unified_df['Biden_position'] - unified_df['Trump_position']) * 0.25
        
        if not appellatives_df.empty:
            appellatives_df['poll_id'] = appellatives_df['poll_id'].astype(str)
            unified_df = unified_df.merge(appellatives_df[['poll_id', 'Trump_label', 'Biden_label']], 
                                          left_on='tweet_id', right_on='poll_id', how='left')
            label_map = {'formal': 1.0, 'informal': -1.0, 'neutral': 0.0}
            unified_df['trump_formal_appellative'] = unified_df['Trump_label'].map(label_map)
            unified_df['biden_formal_appellative'] = unified_df['Biden_label'].map(label_map)
            
        if not leaning_df.empty:
            leaning_df['poll_id'] = leaning_df['poll_id'].astype(str)
            unified_df = unified_df.merge(leaning_df[['poll_id', 'conservative_score', 'liberal_score', 'trump_support_score', 'biden_support_score', 'anti_trump_score', 'anti_biden_score']], 
                                          left_on='tweet_id', right_on='poll_id', how='left')
            unified_df['text_partisan_score'] = unified_df['conservative_score']
            
        unified_df['trump_share'] = unified_df['Trump_percentage'] / 100.0 if 'Trump_percentage' in unified_df else np.nan
        unified_df['biden_share'] = unified_df['Biden_percentage'] / 100.0 if 'Biden_percentage' in unified_df else np.nan

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
    
    def compute_poll_outcome_bias(row):
        trump_share = row.get('trump_share')
        biden_share = row.get('biden_share')
        if pd.notna(trump_share) and pd.notna(biden_share):
            return np.clip(trump_share - biden_share, -1.0, 1.0)
        return np.nan
        
    unified_df['poll_outcome'] = unified_df.apply(compute_poll_outcome_bias, axis=1)

    # Formality bias 
    def compute_formality_bias(row):
        trump_f = row.get('trump_formal_appellative', 0)
        biden_f = row.get('biden_formal_appellative', 0)
        if pd.isna(trump_f): trump_f = 0
        if pd.isna(biden_f): biden_f = 0
        trump_f, biden_f = int(trump_f), int(biden_f)
        return float(np.sign(trump_f - biden_f))

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
        
        # Dichotomize 
        plot_df['marker_dichotomized'] = np.where(plot_df[marker_col] >= median_val, 'High', 'Low')
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=plot_df, 
            x='audience_bias', 
            y='poll_outcome', 
            hue='marker_dichotomized',
            palette={'High': 'tab:red', 'Low': 'tab:blue'},
            alpha=0.6,
            s=50,
            edgecolor='white'
        )
        
        # Add zero lines
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f'Relationship Between Audience Bias and Poll Outcome\nHue: {marker_name} (Dichotomized at Median:{median_val:.3f})')
        plt.xlabel('Audience Bias (-1=Left/Pro-Biden, +1=Right/Pro-Trump)')
        plt.ylabel('Poll Outcome Bias (-1=Biden win, +1=Trump win)')
        plt.legend(title=f'{marker_name}\n(High = Right/Trump-leaning)')
        
        output_path = FIGURES_DIR / f"scatter_{marker_col}_bias.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Saved scatter plot to {output_path}")

@app.command()
def run_all():
    """
    Run all plotting functions to generate complete set of visualizations.
    """
    candidate_order()
    appellatives()
    leaning() 
    bias_relationship_scatter()

if __name__ == "__main__":
    app()
