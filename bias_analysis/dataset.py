"""
Data loading utilities for Twitter poll bias analysis.

This module provides functions to load and parse JSONL files containing
Twitter poll data from multiple sources used in the 2020 US election analysis.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from collections import defaultdict

from loguru import logger
from tqdm import tqdm
import typer

from bias_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def load_jsonl_dataset(file_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load JSONL (JSON Lines) dataset into pandas DataFrame.
    
    Handles malformed JSON lines gracefully by logging warnings and continuing
    to process valid lines. Progress is tracked with tqdm for large files.
    
    Args:
        file_path: Path to the JSONL file
        max_rows: Optional limit on number of rows to load (for testing/debugging)
        
    Returns:
        DataFrame with loaded data, empty DataFrame if no valid data found
    """
    data = []
    logger.info(f"Loading data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path.name}")):
            if max_rows and i >= max_rows:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {i+1} in {file_path}: {e}")
                continue
    
    if not data:
        logger.warning(f"No valid data found in {file_path}")
        return pd.DataFrame()

    logger.success(f"Loaded {len(data)} records from {file_path}")
    return pd.DataFrame(data)

def load_user_score_jsonl(file_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load JSONL files where each line contains a single user ID to score mapping.
    
    These files have the format: {"user_id": score_value} per line.
    This function normalizes them into a DataFrame with 'user_id' and 'score' columns.
    
    Args:
        file_path: Path to the JSONL file with user scores
        max_rows: Optional limit on number of rows to load (for testing/debugging)
        
    Returns:
        DataFrame with columns ['user_id', 'score']
    """
    user_scores = []
    logger.info(f"Loading user scores from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path.name}")):
            if max_rows and i >= max_rows:
                break
            try:
                line_data = json.loads(line.strip())
                # Each line should contain exactly one user_id: score pair
                for user_id, score in line_data.items():
                    user_scores.append({'user_id': user_id, 'score': score})
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {i+1} in {file_path}: {e}")
                continue
    
    if not user_scores:
        logger.warning(f"No valid user scores found in {file_path}")
        return pd.DataFrame(columns=['user_id', 'score'])

    logger.success(f"Loaded {len(user_scores)} user scores from {file_path}")
    return pd.DataFrame(user_scores)

def load_user_demographics_jsonl(file_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load JSONL files where each line contains user ID to demographic data mapping.
    
    These files have the format: {"user_id": {"gender": {...}, "age": {...}, "org": {...}}} per line.
    This function normalizes them into a DataFrame with demographic columns.
    
    Args:
        file_path: Path to the JSONL file with user demographics  
        max_rows: Optional limit on number of rows to load (for testing/debugging)
        
    Returns:
        DataFrame with user demographics data
    """
    user_demographics = []
    logger.info(f"Loading user demographics from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path.name}")):
            if max_rows and i >= max_rows:
                break
            try:
                line_data = json.loads(line.strip())
                # Each line should contain exactly one user_id: demographics pair
                for user_id, demographics in line_data.items():
                    demo_record = {'user_id': user_id}
                    
                    # Extract gender probabilities
                    if 'gender' in demographics:
                        demo_record['gender_male_prob'] = demographics['gender'].get('male', 0)
                        demo_record['gender_female_prob'] = demographics['gender'].get('female', 0)
                    
                    # Extract age probabilities
                    if 'age' in demographics:
                        demo_record['age_18_under_prob'] = demographics['age'].get('<=18', 0)
                        demo_record['age_19_29_prob'] = demographics['age'].get('19-29', 0)
                        demo_record['age_30_39_prob'] = demographics['age'].get('30-39', 0)
                        demo_record['age_40_over_prob'] = demographics['age'].get('>=40', 0)
                    
                    # Extract organization status
                    if 'org' in demographics:
                        demo_record['org_non_org_prob'] = demographics['org'].get('non-org', 0)
                        demo_record['org_is_org_prob'] = demographics['org'].get('is-org', 0)
                    
                    user_demographics.append(demo_record)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {i+1} in {file_path}: {e}")
                continue
    
    if not user_demographics:
        logger.warning(f"No valid user demographics found in {file_path}")
        return pd.DataFrame()

    logger.success(f"Loaded {len(user_demographics)} user demographic records from {file_path}")
    return pd.DataFrame(user_demographics)

def load_polls(max_rows_per_file: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three primary Twitter poll datasets.
    
    Each dataset contains Twitter polls with candidate options and vote percentages
    that will be analyzed for various bias effects and marker extractions.
    
    Args:
        max_rows_per_file: Optional limit on rows per file (useful for development/testing)
        
    Returns:
        Tuple containing (decahose_df, vote_df, voting_df) DataFrames
        
    Raises:
        FileNotFoundError: If any required data file is missing
    """
    logger.info("Loading poll datasets...")
    
    # Define paths to the three core poll datasets
    decahose_input_path = RAW_DATA_DIR / "Decahose/polls.jsonl"
    vote_input_path = RAW_DATA_DIR / "vote/poll-vote-2020.jsonl"
    voting_input_path = RAW_DATA_DIR / "voting/poll-voting-2020.jsonl"
    
    # Validate all required files exist before attempting to load
    for path in [decahose_input_path, vote_input_path, voting_input_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
    
    # Load each dataset with progress tracking
    decahose_df = load_jsonl_dataset(decahose_input_path, max_rows_per_file)
    vote_df = load_jsonl_dataset(vote_input_path, max_rows_per_file)
    voting_df = load_jsonl_dataset(voting_input_path, max_rows_per_file)
    
    logger.success("Data loading completed successfully!")
    
    return decahose_df, vote_df, voting_df

def load_partisanship_scores(max_rows_per_file: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load partisanship scores from all inference subdirectories.
    
    Partisanship scores indicate political leaning of users across different datasets.
    Higher scores typically indicate stronger partisan alignment.
    
    Args:
        max_rows_per_file: Optional limit on rows per file (useful for development/testing)
        
    Returns:
        Tuple containing (decahose_partisanship_df, vote_partisanship_df, voting_partisanship_df) DataFrames
        Each DataFrame has columns ['user_id', 'score']
        
    Raises:
        FileNotFoundError: If any required partisanship data file is missing
    """
    logger.info("Loading partisanship scores datasets...")
    
    decahose_partisan_path = RAW_DATA_DIR / "Decahose/inference/partisanship_scores_final.jsonl"
    vote_partisan_path = RAW_DATA_DIR / "vote/inference/partisanship_scores_all_users.jsonl"
    voting_partisan_path = RAW_DATA_DIR / "voting/inference/partisan-voting-2020.jsonl"
    
    # Validate all required files exist before attempting to load
    for path in [decahose_partisan_path, vote_partisan_path, voting_partisan_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required partisanship data file not found: {path}")
    
    decahose_partisanship_df = load_user_score_jsonl(decahose_partisan_path, max_rows_per_file)
    vote_partisanship_df = load_user_score_jsonl(vote_partisan_path, max_rows_per_file)
    voting_partisanship_df = load_user_score_jsonl(voting_partisan_path, max_rows_per_file)
    
    logger.success("Partisanship scores loading completed successfully!")
    
    return decahose_partisanship_df, vote_partisanship_df, voting_partisanship_df

def load_demographic_data(max_rows_per_file: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load M3 demographic inference data from all inference subdirectories.
    
    Contains demographic predictions (gender, age, organization status) for users
    generated using the M3 (Multi-task, Multi-domain, Multi-lingual) model.
    
    Args:
        max_rows_per_file: Optional limit on rows per file (useful for development/testing)
        
    Returns:
        Tuple containing (decahose_demographic_df, vote_demographic_df, voting_demographic_df) DataFrames
        Each DataFrame has demographic probability columns for each user
        
    Raises:
        FileNotFoundError: If any required demographic data file is missing
    """
    logger.info("Loading demographic inference datasets...")
    
    decahose_demographic_path = RAW_DATA_DIR / "Decahose/inference/m3inf_output_final.jsonl"
    vote_demographic_path = RAW_DATA_DIR / "vote/inference/m3inf_output_all_users.jsonl"
    voting_demographic_path = RAW_DATA_DIR / "voting/inference/m3inf-voting-2020.jsonl"
    
    # Validate all required files exist before attempting to load
    for path in [decahose_demographic_path, vote_demographic_path, voting_demographic_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required demographic data file not found: {path}")
    
    decahose_demographic_df = load_user_demographics_jsonl(decahose_demographic_path, max_rows_per_file)
    vote_demographic_df = load_user_demographics_jsonl(vote_demographic_path, max_rows_per_file)
    voting_demographic_df = load_user_demographics_jsonl(voting_demographic_path, max_rows_per_file)
    
    logger.success("Demographic inference data loading completed successfully!")
    
    return decahose_demographic_df, vote_demographic_df, voting_demographic_df

def load_engagement_data(max_rows_per_file: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load retweeter and favoriter engagement data from all datasets.
    
    Combines user engagement data (retweets and favorites) from all three data sources
    to enable audience analysis for political bias research.
    
    Args:
        max_rows_per_file: Optional limit on rows per file (useful for development/testing)
        
    Returns:
        Tuple containing (all_retweeters_df, all_favoriters_df) DataFrames
        Each DataFrame contains user profile data from engagement actions
        
    Raises:
        FileNotFoundError: If any required engagement data file is missing
    """
    logger.info("Loading engagement data (retweeters and favoriters)...")
    
    retweeter_paths = [
        RAW_DATA_DIR / "Decahose/retweeters.jsonl",
        RAW_DATA_DIR / "vote/retweet-vote-2020.jsonl", 
        RAW_DATA_DIR / "voting/retweet-voting-2020.jsonl"
    ]
    
    favoriter_paths = [
        RAW_DATA_DIR / "Decahose/favoriters.jsonl",
        RAW_DATA_DIR / "vote/favorite-vote-2020.jsonl",
        RAW_DATA_DIR / "voting/favoriters-voting-2020.jsonl"
    ]
    
    # Validate all required files exist
    all_paths = retweeter_paths + favoriter_paths
    for path in all_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required engagement data file not found: {path}")
    
    # Load and combine retweeter data
    retweeter_dfs = []
    for path in retweeter_paths:
        df = load_jsonl_dataset(path, max_rows_per_file)
        if not df.empty:
            df['data_source'] = path.parent.name
            retweeter_dfs.append(df)
    
    # Load and combine favoriter data  
    favoriter_dfs = []
    for path in favoriter_paths:
        df = load_jsonl_dataset(path, max_rows_per_file)
        if not df.empty:
            df['data_source'] = path.parent.name
            favoriter_dfs.append(df)
    
    all_retweeters_df = pd.concat(retweeter_dfs, ignore_index=True) if retweeter_dfs else pd.DataFrame()
    all_favoriters_df = pd.concat(favoriter_dfs, ignore_index=True) if favoriter_dfs else pd.DataFrame()
    
    logger.success(f"Loaded {len(all_retweeters_df)} retweeter records and {len(all_favoriters_df)} favoriter records")
    
    return all_retweeters_df, all_favoriters_df

def create_unified_poll_dataset(max_rows_per_file: Optional[int] = None, drop_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Create comprehensive poll-level dataset combining all data sources and processed markers.
    
    Produces a single table where each row represents one poll tweet with:
    - Basic poll metadata (tweet_id, author_id, created_at, poll options/votes)
    - Processed bias markers (candidate_order, appellatives, political_leaning with 6 dimensions)
    - Political categorization ('pro-trump', 'neutral', 'pro-biden') using sophisticated logic
    - Numerical partisan score combining all 6 political dimensions
    - Author characteristics (partisanship, demographics, followers)
    - Audience metrics (partisanship statistics from retweeters/favoriters)
    - Poll outcome variables (vote shares, entropy, winner)
    
    Political scores included:
    - conservative_score, liberal_score (ideological alignment)
    - trump_support_score, biden_support_score (candidate support)
    - anti_trump_score, anti_biden_score (candidate opposition)
    
    Args:
        max_rows_per_file: Optional limit on rows per file (useful for development/testing)
        drop_columns: Optional list of column names to drop from final dataset to reduce size
        
    Returns:
        DataFrame with unified poll-level data ready for correlation analysis
        
    Raises:
        FileNotFoundError: If any required processed marker files are missing
    """
    logger.info("Creating unified poll dataset...")
    
    # Step 1: Verify all processed marker files exist
    processed_files = {
        'candidate_order': PROCESSED_DATA_DIR / "candidate_order_features.csv",
        'appellatives': PROCESSED_DATA_DIR / "formal_informal_appellatives.csv", 
        'political_leaning': PROCESSED_DATA_DIR / "political_leaning_features.csv"
    }
    
    for marker_name, file_path in processed_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Required processed marker file not found: {file_path}")
    
    # Step 2: Load processed bias markers from CSV files
    logger.info("Loading processed bias markers...")
    candidate_order_df = pd.read_csv(processed_files['candidate_order'])
    appellatives_df = pd.read_csv(processed_files['appellatives'])
    political_leaning_df = pd.read_csv(processed_files['political_leaning'])
    
    # Step 3: Load raw poll data from all three sources
    decahose_polls, vote_polls, voting_polls = load_polls(max_rows_per_file)
    
    # Step 4: Load auxiliary datasets (partisanship, demographics, engagement)
    logger.info("Loading auxiliary data sources...")
    partisanship_dfs = load_partisanship_scores(max_rows_per_file)
    demographic_dfs = load_demographic_data(max_rows_per_file)
    retweeters_df, favoriters_df = load_engagement_data(max_rows_per_file)
    
    # Step 5: Merge partisanship scores from Decahose, vote, and voting datasets
    all_partisanship_df = pd.concat([
        partisanship_dfs[0].assign(source='decahose'),
        partisanship_dfs[1].assign(source='vote'), 
        partisanship_dfs[2].assign(source='voting')
    ], ignore_index=True)
    
    # Ensure user_id is string type for consistent merging
    all_partisanship_df['user_id'] = all_partisanship_df['user_id'].astype(str)
    # Ensure score is numeric type for z-score calculation
    all_partisanship_df['score'] = pd.to_numeric(all_partisanship_df['score'], errors='coerce')
    
    # Remove duplicates, keeping first occurrence
    all_partisanship_df = all_partisanship_df.drop_duplicates(subset=['user_id'], keep='first')
    
    # Step 6: Merge demographic data from all sources
    all_demographics_df = pd.concat([
        demographic_dfs[0].assign(source='decahose'),
        demographic_dfs[1].assign(source='vote'),
        demographic_dfs[2].assign(source='voting')
    ], ignore_index=True)
    
    all_demographics_df['user_id'] = all_demographics_df['user_id'].astype(str)
    all_demographics_df = all_demographics_df.drop_duplicates(subset=['user_id'], keep='first')
    
    # Step 7: Extract poll metadata from raw tweet data
    all_polls = []
    seen_poll_ids = set()
    
    for polls_df, source_name in [(decahose_polls, 'decahose'), (vote_polls, 'vote'), (voting_polls, 'voting')]:
        for _, row in polls_df.iterrows():
            poll_id = str(row.get('id', ''))
            
            if poll_id in seen_poll_ids:
                continue
            seen_poll_ids.add(poll_id)
            
            if not isinstance(row.get('entities'), dict):
                continue
            entities = row['entities']
            if 'polls' not in entities or not entities['polls']:
                continue
                
            poll = entities['polls'][0]
            options = poll.get('options', [])
            if not options:
                continue
            
            poll_record = {
                'tweet_id': poll_id,
                'author_id': str(row.get('user', {}).get('id_str', '')),
                'created_at': row.get('created_at', ''),
                'tweet_text': row.get('text', ''),
                'data_source': source_name,
                'duration_minutes': poll.get('duration_minutes', None),
                'n_options': len(options),
                'total_votes': sum(option.get('votes', 0) for option in options)
            }
            
            poll_options = []
            for option in options:
                option_text = option.get('label', option.get('text', '')).strip()
                votes = option.get('votes', 0)
                position = option.get('position', 0)
                poll_options.append({'position': position, 'label': option_text, 'votes': votes})
            
            # Trump/Biden-specific vote data will come from candidate_order_features.csv
            # to ensure consistency with the smart_candidate_match logic used in features.py
            
            total_votes = poll_record['total_votes'] if poll_record['total_votes'] > 0 else 1
            
            # Shannon entropy: H = -Σ(p_i * log2(p_i)) measures poll outcome uncertainty
            vote_shares = [opt['votes'] / total_votes for opt in poll_options if opt['votes'] > 0]
            if vote_shares:
                poll_record['entropy_votes'] = -sum(p * np.log2(p) for p in vote_shares if p > 0)
            else:
                poll_record['entropy_votes'] = 0
            
            poll_record['poll_options'] = str(poll_options)
            
            all_polls.append(poll_record)
    
    unified_df = pd.DataFrame(all_polls)
    logger.info(f"Processed {len(unified_df)} unique polls from all sources")
    
    unified_df['tweet_id'] = unified_df['tweet_id'].astype(str)
    unified_df['author_id'] = unified_df['author_id'].astype(str)
    
    # Step 8: Merge bias markers and compute derived poll outcome metrics
    logger.info("Merging processed bias markers...")
    
    candidate_order_df['poll_id'] = candidate_order_df['poll_id'].astype(str)
    appellatives_df['poll_id'] = appellatives_df['poll_id'].astype(str) 
    political_leaning_df['poll_id'] = political_leaning_df['poll_id'].astype(str)
    
    unified_df = unified_df.merge(
        candidate_order_df[['poll_id', 'Trump_position', 'Biden_position', 'Trump_votes', 'Biden_votes', 'Trump_percentage', 'Biden_percentage']],
        left_on='tweet_id', right_on='poll_id', how='left'
    )
    
    # Candidate order bias marker: position difference scaled to [-1, +1]
    # Formula: (Biden_pos - Trump_pos) * 0.25
    # -1 = Biden first (left-leaning order), +1 = Trump first (right-leaning order)
    # Examples: Biden 1st, Trump 3rd → (1-3)*0.25 = -0.5 | Trump 1st, Biden 3rd → (3-1)*0.25 = +0.5
    unified_df['candidate_order'] = np.where(
        (unified_df['Biden_position'].notna()) & (unified_df['Trump_position'].notna()),
        (unified_df['Biden_position'] - unified_df['Trump_position']) * 0.25,
        np.nan
    )
    
    # Poll outcome metrics: convert percentages to proportions [0, 1]
    unified_df['trump_share'] = unified_df['Trump_percentage'].fillna(0) / 100
    unified_df['biden_share'] = unified_df['Biden_percentage'].fillna(0) / 100
    unified_df['other_share'] = np.maximum(0, 1 - unified_df['trump_share'] - unified_df['biden_share'])
    
    unified_df['winning_share'] = np.maximum.reduce([
        unified_df['trump_share'], 
        unified_df['biden_share'], 
        unified_df['other_share']
    ])
    
    unified_df = unified_df.merge(
        appellatives_df[['poll_id', 'Trump_label', 'Biden_label']],
        left_on='tweet_id', right_on='poll_id', how='left', suffixes=('', '_appellative')
    )
    
    # Appellative bias marker: formal titles (President, Mr.) vs informal (first names, nicknames)
    unified_df['trump_formal_appellative'] = np.where(
        unified_df['Trump_label'] == 'formal', 1,
        np.where(unified_df['Trump_label'].notna(), 0, np.nan)
    )
    unified_df['biden_formal_appellative'] = np.where(
        unified_df['Biden_label'] == 'formal', 1,
        np.where(unified_df['Biden_label'].notna(), 0, np.nan)
    )
    
    unified_df = unified_df.merge(
        political_leaning_df[['poll_id', 'conservative_score', 'liberal_score', 'trump_support_score', 'biden_support_score', 'anti_trump_score', 'anti_biden_score']],
        left_on='tweet_id', right_on='poll_id', how='left', suffixes=('', '_political')
    )
    
    def categorize_political_leaning(row):
        """
        Categorize poll political leaning based on sentiment scores.
        
        Thresholds match visualization logic in plots.py:
        - Pro-Trump: trump_support OR anti_biden >0.5, OR (conservative >0.5 AND liberal <0.5)
        - Pro-Biden: biden_support OR anti_trump >0.5, OR (liberal >0.5 AND conservative <0.5)
        - Neutral: mixed signals or all scores below thresholds
        """
        trump_support = row['trump_support_score'] if pd.notna(row['trump_support_score']) else 0
        biden_support = row['biden_support_score'] if pd.notna(row['biden_support_score']) else 0
        anti_trump = row['anti_trump_score'] if pd.notna(row['anti_trump_score']) else 0
        anti_biden = row['anti_biden_score'] if pd.notna(row['anti_biden_score']) else 0
        conservative = row['conservative_score'] if pd.notna(row['conservative_score']) else 0
        liberal = row['liberal_score'] if pd.notna(row['liberal_score']) else 0
        
        trump_indicators = [
            trump_support > 0.5,
            anti_biden > 0.5,
            (conservative > 0.5 and liberal < 0.5)
        ]
        
        biden_indicators = [
            biden_support > 0.5, 
            anti_trump > 0.5,
            (liberal > 0.5 and conservative < 0.5)
        ]
        
        if any(trump_indicators) and not any(biden_indicators):
            return 'pro-trump'
        elif any(biden_indicators) and not any(trump_indicators):
            return 'pro-biden'
        else:
            return 'neutral'
    
    unified_df['political_leaning'] = unified_df.apply(categorize_political_leaning, axis=1)
    
    # Continuous partisan score: aggregate all 6 BART-MNLI dimensions into single [-1, +1] metric
    # Formula: mean(pro-Trump scores) - mean(pro-Biden scores)
    # +1 = strongly pro-Trump, -1 = strongly pro-Biden, 0 = neutral/balanced
    unified_df['text_partisan_score'] = (
        (unified_df['trump_support_score'].fillna(0) + unified_df['anti_biden_score'].fillna(0) + unified_df['conservative_score'].fillna(0)) -
        (unified_df['biden_support_score'].fillna(0) + unified_df['anti_trump_score'].fillna(0) + unified_df['liberal_score'].fillna(0))
    ) / 3
    
    # Step 9: Attach poll author characteristics
    logger.info("Adding author characteristics...")
    
    unified_df = unified_df.merge(
        all_partisanship_df[['user_id', 'score']].rename(columns={'score': 'author_partisanship_raw'}),
        left_on='author_id', right_on='user_id', how='left'
    )
    
    # Z-score normalization: (x - μ) / σ transforms partisanship to standard deviations from mean
    # Enables comparison across datasets with different score scales
    partisanship_raw = unified_df['author_partisanship_raw']
    partisanship_mean = partisanship_raw.mean()
    partisanship_std = partisanship_raw.std()
    
    # Only standardize if we have valid data and non-zero standard deviation
    if pd.notna(partisanship_mean) and pd.notna(partisanship_std) and partisanship_std > 0:
        unified_df['author_partisanship'] = (partisanship_raw - partisanship_mean) / partisanship_std
    else:
        unified_df['author_partisanship'] = partisanship_raw
    
    unified_df = unified_df.merge(
        all_demographics_df[['user_id', 'org_is_org_prob']].rename(columns={'org_is_org_prob': 'author_org_prob'}),
        left_on='author_id', right_on='user_id', how='left', suffixes=('', '_demo')
    )
    
    # Step 10: Compute audience partisanship metrics from engagement data
    logger.info("Calculating audience metrics...")
    
    # Build mapping: tweet_id → {unique user_ids, their partisanship scores}
    # Ensures each user counted once per tweet even if both retweeted and favorited
    audience_engagement = defaultdict(lambda: {'users': set(), 'partisanship_scores': []})
    
    # Process retweeters: JSONL format is {"tweet_id": [user_obj1, user_obj2, ...]}
    if not retweeters_df.empty:
        for _, row in retweeters_df.iterrows():
            for tweet_id, users_list in row.items():
                if isinstance(users_list, list):
                    tweet_id = str(tweet_id)
                    for user_obj in users_list:
                        if isinstance(user_obj, dict):
                            user_id = str(user_obj.get('id', ''))
                            if user_id and user_id not in audience_engagement[tweet_id]['users']:
                                user_partisanship = all_partisanship_df[all_partisanship_df['user_id'] == user_id]
                                if not user_partisanship.empty:
                                    audience_engagement[tweet_id]['users'].add(user_id)
                                    audience_engagement[tweet_id]['partisanship_scores'].append(user_partisanship['score'].iloc[0])
    
    # Process favoriters: same data structure, deduplicate against retweeters
    if not favoriters_df.empty:
        for _, row in favoriters_df.iterrows():
            for tweet_id, users_list in row.items():
                if isinstance(users_list, list):
                    tweet_id = str(tweet_id)
                    for user_obj in users_list:
                        if isinstance(user_obj, dict):
                            user_id = str(user_obj.get('id', ''))
                            if user_id and user_id not in audience_engagement[tweet_id]['users']:
                                user_partisanship = all_partisanship_df[all_partisanship_df['user_id'] == user_id]
                                if not user_partisanship.empty:
                                    audience_engagement[tweet_id]['users'].add(user_id)
                                    audience_engagement[tweet_id]['partisanship_scores'].append(user_partisanship['score'].iloc[0])
    
    # Aggregate statistics: mean/median partisanship and sample size
    audience_stats = []
    for tweet_id in unified_df['tweet_id']:
        scores = audience_engagement.get(tweet_id, {}).get('partisanship_scores', [])
        if scores:
            audience_stats.append({
                'tweet_id': tweet_id,
                'audience_mean_partisanship': np.mean(scores),
                'audience_median_partisanship': np.median(scores),
                'audience_n_distinct_users': len(scores)
            })
        else:
            audience_stats.append({
                'tweet_id': tweet_id,
                'audience_mean_partisanship': np.nan,
                'audience_median_partisanship': np.nan,
                'audience_n_distinct_users': 0
            })
    
    audience_df = pd.DataFrame(audience_stats)
    unified_df = unified_df.merge(audience_df, on='tweet_id', how='left')
    
    # Step 11: Clean up redundant columns and apply optional size reduction
    cols_to_drop = [col for col in unified_df.columns if col.endswith('_y') or 'poll_id' in col]
    unified_df = unified_df.drop(columns=cols_to_drop, errors='ignore')
    
    if drop_columns:
        existing_drop_cols = [col for col in drop_columns if col in unified_df.columns]
        missing_cols = [col for col in drop_columns if col not in unified_df.columns]
        
        if existing_drop_cols:
            unified_df = unified_df.drop(columns=existing_drop_cols)
            logger.info(f"Dropped {len(existing_drop_cols)} columns: {existing_drop_cols}")
        
        if missing_cols:
            logger.warning(f"Requested columns not found (already absent): {missing_cols}")
    
    logger.success(f"Created unified dataset with {len(unified_df)} polls and {len(unified_df.columns)} features")
    
    return unified_df

if __name__ == "__main__":
    logger.warning("dataset.py is not intended to be run directly. Use features.py instead.")
