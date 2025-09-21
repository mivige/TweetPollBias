"""
Feature extraction for Twitter poll bias analysis.

This module processes raw Twitter poll data to extract features related to 
candidate positioning and voting patterns. The primary focus is analyzing
how candidate order in poll options affects voting outcomes.
"""

from pathlib import Path
import pandas as pd
import json
from collections import defaultdict

from loguru import logger
from tqdm import tqdm
import typer

from bias_analysis.config import PROCESSED_DATA_DIR
from bias_analysis.dataset import load_for_candidate_order

app = typer.Typer()


@app.command()
def candidate_order():
    """
    Extract candidate position and voting data from Twitter polls.
    
    Processes Twitter poll data to create a structured dataset where each row represents
    one poll with candidate positions and vote percentages. This enables analysis of
    whether candidate order in poll options influences voting outcomes.
    
    Key processing steps:
    1. Normalize candidate names to handle variations (Trump/TRUMP/Donald Trump)
    2. Extract poll option positions and vote counts
    3. Calculate vote percentages 
    4. Create one-row-per-poll structure for analysis
    
    Output CSV contains columns for each major candidate:
    - {candidate}_position: Position in poll options (1, 2, 3, 4)
    - {candidate}_votes: Raw vote count  
    - {candidate}_percentage: Vote percentage
    """
    logger.info("Starting candidate order feature extraction...")
    
    # Define candidate name variations for robust matching
    # This handles the many ways candidates are referenced in Twitter polls
    main_candidates = {
        'Trump': ['Trump', 'trump', 'TRUMP', 'Donald Trump', 'donald trump', 
                 'President Trump', 'Donald J. Trump', 'Donald J Trump'],
        'Biden': ['Biden', 'biden', 'BIDEN', 'Joe Biden', 'joe biden', 
                 'President Biden', 'Joe Biden (Democrat)', 'Sleepy Joe', 'sleepy joe'],
        'Other': []  # Catch-all for other candidates not in main analysis
    }
    
    # Create efficient lookup for candidate name normalization
    candidate_mapping = {}
    for main_name, variations in main_candidates.items():
        for variation in variations:
            candidate_mapping[variation.strip()] = main_name
    
    def smart_candidate_match(candidate_text):
        """
        Enhanced candidate matching that handles variations with emojis, prefixes, etc.
        
        This function attempts to match candidate names even when they include:
        - Emojis (🇺🇸, 💙, 🤮, etc.)
        - Common prefixes ("vote for", "voting for", "elect", etc.)
        - Common suffixes ("2020", "for president", etc.)
        - Case variations and extra whitespace
        
        Args:
            candidate_text (str): Raw candidate text from poll option
            
        Returns:
            str: Normalized candidate name ('Trump', 'Biden', or 'Other')
        """
        if not candidate_text or not isinstance(candidate_text, str):
            return 'Other'
        
        # First try exact match (fastest path)
        normalized_text = candidate_text.strip()
        if normalized_text in candidate_mapping:
            return candidate_mapping[normalized_text]
        
        # Clean the text for fuzzy matching
        import re
        
        # Remove emojis and special characters but keep letters, numbers, and basic punctuation
        cleaned_text = re.sub(r'[^\w\s\-\.\(\)\/]', ' ', normalized_text)
        
        # Remove common prefixes that don't affect candidate identity
        prefixes_to_remove = [
            r'\b(?:vote|voting|elect|choose|pick|select|support|go|team|go with|pick)\s+(?:for\s+)?',
            r'\b(?:president|pres\.?|mr\.?|senator|sen\.?|vice president|vp)\s+',
            r'\b(?:democratic|democrat|republican|gop|dem)\s+',
            r'\b(?:candidate|nominee)\s+',
            r'\b(?:i choose|i pick|i vote|i support|i want|i prefer)\s+',
            r'\b(?:definitely|probably|maybe|likely)\s+',
            r'\b(?:gonna vote|will vote|voting)\s+(?:for\s+)?'
        ]
        
        for prefix_pattern in prefixes_to_remove:
            cleaned_text = re.sub(prefix_pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove common suffixes
        suffixes_to_remove = [
            r'\s+(?:2020|2024|for president|for pres|presidency|administration|admin)$',
            r'\s+(?:ticket|campaign|rally|supporters?)$',
            r'\s+(?:wins?|victory|loses?|defeat)$',
            r'\s+(?:\(democrat\)|\(republican\)|\(gop\)|\(dem\)|\(r\)|\(d\))$',
            r'\s+(?:again|still|now|then|too|also)$'
        ]
        
        for suffix_pattern in suffixes_to_remove:
            cleaned_text = re.sub(suffix_pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned_text = ' '.join(cleaned_text.split())
        
        # Try exact match after cleaning
        if cleaned_text in candidate_mapping:
            return candidate_mapping[cleaned_text]
        
        # Try case-insensitive partial matching for core candidate names
        cleaned_lower = cleaned_text.lower()
        
        # Define core name patterns for fuzzy matching
        trump_patterns = [
            r'\btrump\b', r'\bdonald\b.*\btrump\b', r'\btrump\b.*\bdonald\b',
            r'\bpresident\s+trump\b', r'\bdonald\s+j\.?\s+trump\b'
        ]
        
        biden_patterns = [
            r'\bbiden\b', r'\bjoe\b.*\bbiden\b', r'\bbiden\b.*\bjoe\b',
            r'\bpresident\s+biden\b', r'\bjoseph\s+biden\b'
        ]
        
        # Check Trump patterns
        for pattern in trump_patterns:
            if re.search(pattern, cleaned_lower):
                return 'Trump'
        
        # Check Biden patterns  
        for pattern in biden_patterns:
            if re.search(pattern, cleaned_lower):
                return 'Biden'
        
        # If no match found, return 'Other'
        return 'Other'
    
    # Load raw poll data from three Twitter data sources
    decahose_df, vote_df, voting_df = load_for_candidate_order()
    
    # Process each dataset maintaining source attribution
    all_dataframes = [
        (decahose_df, "decahose"),
        (vote_df, "vote"), 
        (voting_df, "voting")
    ]
    
    candidate_order_data = []
    
    for df, dataset_name in all_dataframes:
        logger.info(f"Processing {dataset_name} dataset with {len(df)} records...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
            # Validate row contains poll data structure
            if not isinstance(row.get('entities'), dict):
                continue
                
            entities = row['entities']
            if 'polls' not in entities or not entities['polls']:
                continue
                
            # Extract first poll from tweet (tweets can contain multiple polls)
            poll = entities['polls'][0]
            options = poll.get('options', [])
            
            if not options:
                continue
            
            # Calculate total votes for percentage normalization
            total_votes = sum(option.get('votes', 0) for option in options)
            if total_votes == 0:
                total_votes = 1  # Prevent division by zero in percentage calculation
            
            # Initialize poll record with metadata
            poll_record = {
                'poll_id': str(row.get('id', '')),
                'tweet_text': row.get('text', ''),
                'dataset_source': dataset_name,
                'created_at': row.get('created_at', ''),
                'user_id': str(row.get('user', {}).get('id', '')),
                'total_votes': total_votes,
                'num_candidates': len(options)
            }
            
            # Initialize candidate data columns
            for candidate in main_candidates.keys():
                poll_record[f'{candidate}_position'] = None
                poll_record[f'{candidate}_votes'] = None
                poll_record[f'{candidate}_percentage'] = None
            
            # Extract position and vote data for each poll option
            for option in options:
                candidate_name = option.get('label', option.get('text', '')).strip()
                position = option.get('position', 0)
                votes = option.get('votes', 0)
                vote_percentage = (votes / total_votes) * 100 if total_votes > 0 else 0
                
                # Use smart matching to handle variations with emojis, prefixes, etc.
                normalized_name = smart_candidate_match(candidate_name)
                
                # Store data for first occurrence of candidate (handles duplicates)
                if poll_record[f'{normalized_name}_position'] is None:
                    poll_record[f'{normalized_name}_position'] = position
                    poll_record[f'{normalized_name}_votes'] = votes
                    poll_record[f'{normalized_name}_percentage'] = vote_percentage
            
            candidate_order_data.append(poll_record)
    
    # Convert to DataFrame for analysis
    result_df = pd.DataFrame(candidate_order_data)
    
    if result_df.empty:
        logger.warning("No poll data found in any dataset!")
        return
    
    # Save processed features
    output_path = PROCESSED_DATA_DIR / "candidate_order_features.csv"
    result_df.to_csv(output_path, index=False)
    
    logger.success(f"Candidate order features saved to {output_path}")
    logger.info(f"Generated {len(result_df)} poll records")
    logger.info(f"Columns created: {len(result_df.columns)}")
    
    # Report coverage statistics for main candidates
    for candidate in main_candidates.keys():
        count = result_df[f'{candidate}_position'].notna().sum()
        logger.info(f"Polls with {candidate}: {count}")
    
    return result_df

@app.command()
def political_leaning():
    """
    Computes the political leaning of the text based on the presence of certain keywords.
    """
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Political leaning feature order:")
    features = ["leaning_a", "leaning_b", "leaning_c"]
    for feature in features:
        logger.info(f"- {feature}")
    # -----------------------------------------

@app.command()
def formal_vs_informal():
    """
    Extracts the formality of the appellatives used in polls options and tweet texts through Named Entity Recognition (to be manually labeled).
    """
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Formality feature order:")
    features = ["formal_a", "formal_b", "formal_c"]
    for feature in features:
        logger.info(f"- {feature}")
    # -----------------------------------------

@app.command()
def pearson_correlation():
    """
    Computes the Pearson correlation coefficient between author, audience and poll outcome bias.
    """
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Pearson correlation coefficients:")
    correlations = {"feature_a": 0.5, "feature_b": -0.3, "feature_c": 0.1}
    for feature, coeff in correlations.items():
        logger.info(f"- {feature}: {coeff}")
    # -----------------------------------------

if __name__ == "__main__":
    app()
