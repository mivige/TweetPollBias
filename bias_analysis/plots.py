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
    
    Creates dual-panel visualization showing how poll position (1st, 2nd, 3rd, 4th)
    correlates with vote percentage for Trump and Biden. Uses multiple visual layers:
    - Violin plots: Show distribution density at each position
    - Scatter points: Individual poll results with jitter for clarity
    - Statistical markers: Mean with error bars, median markers
    - Overall median line: Reference line for comparison
    
    Saves both the plot and detailed statistics to files for analysis.
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
        
        Combines violin plots (distribution), jittered scatter points (individual polls),
        statistical markers (means/medians), and reference lines for comprehensive
        visualization of position bias effects.
        
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
            
        # Add horizontal jitter to scatter points for better visibility
        x_base = plot_df_nonzero['position'].values.astype(float)
        jitter = np.random.normal(loc=0, scale=0.08, size=len(plot_df_nonzero))
        x_jittered = x_base + jitter
        
        # Background violin plots show distribution density at each position
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
        
        # Calculate position-wise statistics for overlays
        grp = plot_df_nonzero.groupby('position')['percentage']
        medians = grp.median()
        means = grp.mean()
        sems = grp.sem().fillna(0)  # Standard error of mean
        
        x_positions = sorted(plot_df_nonzero['position'].unique())
        
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
        
        # Reference line: overall median performance for context
        overall_median = plot_df_nonzero['percentage'].median()
        ax.axhline(overall_median, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        # ax.text(0.1, overall_median + 2, f'Overall Median ({overall_median:.1f}%)', color='gray', fontsize=9)
        
        # Axis labels and styling
        ax.set_xlabel('Poll Position')
        ax.set_ylabel('Vote Percentage (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Configure x-axis to show all positions clearly
        max_position = plot_df_nonzero['position'].max()
        min_position = plot_df_nonzero['position'].min()
        ax.set_xlim(min_position - 0.5, max_position + 0.5)
        
        # Explicitly set tick positions and labels
        all_positions = sorted(plot_df_nonzero['position'].unique())
        ax.set_xticks(all_positions)
        ax.set_xticklabels([str(int(pos)) for pos in all_positions])
        
        # Y-axis covers full percentage range with padding
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
    
    Creates dual-panel visualization showing how appellative formality (formal, neutral, informal)
    correlates with vote percentage for Trump and Biden. Uses the same visual design as 
    candidate_order plots:
    - Violin plots: Show distribution density for each formality level
    - Scatter points: Individual poll results with jitter for clarity
    - Statistical markers: Mean with error bars, median markers
    - Overall median line: Reference line for comparison
    
    Analyzes whether formal appellatives (e.g., "President Trump") vs informal ones
    (e.g., "Sleepy Joe") correlate with different voting patterns.
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
        
        Similar to position bias plots but with formality categories on x-axis.
        Combines violin plots (distribution), jittered scatter points (individual polls),
        statistical markers (means/medians), and reference lines for comprehensive
        visualization of appellative formality effects.
        
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
        
        # Background violin plots show distribution density at each formality level
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
        
        # Reference line: overall median performance for context
        overall_median = plot_df_nonzero['percentage'].median()
        ax.axhline(overall_median, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        # Axis labels and styling
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
def run_all():
    """
    Run all plotting functions to generate complete set of visualizations.
    """
    candidate_order()
    appellatives() 

if __name__ == "__main__":
    app()
