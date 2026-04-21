"""
Data loading utilities for Twitter poll bias analysis.

This module provides functions to load and parse JSONL files containing
Twitter poll data from multiple sources. All election-specific file paths
are resolved from bias_analysis.election_configs so the same loader works
across different elections.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from collections import defaultdict

from loguru import logger
from tqdm import tqdm
import typer

from bias_analysis.config import get_election_paths
from bias_analysis.election_configs import get_election_config

app = typer.Typer()

# Default election code used when no election is specified.
DEFAULT_ELECTION = "us20"


def load_jsonl_dataset(file_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load JSONL (JSON Lines) dataset into a pandas DataFrame.

    Args:
        file_path: Path to the JSONL file.
        max_rows: Optional limit on number of rows to load.

    Returns:
        DataFrame with loaded data. Empty if none found.
    """
    data = []
    logger.info(f"Loading data from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path.name}")):
            if isinstance(max_rows, int) and i >= max_rows:
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
    Load JSONL files mapping user IDs to scores.

    Args:
        file_path: Path to the JSONL file.
        max_rows: Optional limit on number of rows to load.

    Returns:
        DataFrame with ['user_id', 'score'] columns.
    """
    user_scores = []
    logger.info(f"Loading user scores from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path.name}")):
            if isinstance(max_rows, int) and i >= max_rows:
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
    Load JSONL files mapping user IDs to demographic data.

    Args:
        file_path: Path to the JSONL file.
        max_rows: Optional limit on number of rows to load.

    Returns:
        DataFrame with demographic columns.
    """
    user_demographics = []
    logger.info(f"Loading user demographics from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path.name}")):
            if isinstance(max_rows, int) and i >= max_rows:
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
                        demo_record['age_under_29_prob'] = demographics['age'].get('<=18', 0) + demographics['age'].get('19-29', 0)
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


def load_engagement_jsonl(file_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load JSONL files containing engagement data (retweeters/favoriters) 
    in the format {"tweet_id": [user_dicts...]}.
    """
    records = []
    logger.info(f"Loading engagement data from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path.name}")):
            if isinstance(max_rows, int) and i >= max_rows:
                break
            try:
                line_data = json.loads(line.strip())
                for tweet_id, users_list in line_data.items():
                    records.append({'tweet_id': tweet_id, 'users_list': users_list})
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {i+1} in {file_path}: {e}")
                continue

    if not records:
        logger.warning(f"No valid engagement data found in {file_path}")
        return pd.DataFrame(columns=['tweet_id', 'users_list'])

    logger.success(f"Loaded {len(records)} engagement records from {file_path}")
    return pd.DataFrame(records)


def _resolve_paths(raw_dir: Path, relative_paths: List[str]) -> List[Path]:
    """Resolve a list of relative paths against a raw data directory."""
    return [raw_dir / rp for rp in relative_paths]


def _validate_paths(paths: List[Path], label: str) -> None:
    """Raise FileNotFoundError if any path does not exist."""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required {label} data file not found: {path}")


def load_polls(
    election: str = DEFAULT_ELECTION,
    max_rows_per_file: Optional[int] = None,
) -> Tuple[pd.DataFrame, ...]:
    """
    Load the core Twitter poll datasets for the given election.

    Args:
        election: Election code (e.g. "us20").
        max_rows_per_file: Optional limit on rows per file.

    Returns:
        Tuple of DataFrames, one per data source.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    raw_dir = paths.raw_dir

    poll_paths = _resolve_paths(raw_dir, ecfg["raw_data_paths"]["polls"])
    _validate_paths(poll_paths, "poll")

    logger.info("Loading poll datasets...")
    dfs = tuple(load_jsonl_dataset(p, max_rows_per_file) for p in poll_paths)
    logger.success("Data loading completed successfully!")
    return dfs


def load_partisanship_scores(
    election: str = DEFAULT_ELECTION,
    max_rows_per_file: Optional[int] = None,
) -> Tuple[pd.DataFrame, ...]:
    """
    Load partisanship scores for all datasets of the given election.

    Args:
        election: Election code (e.g. "us20").
        max_rows_per_file: Optional limit on rows per file.

    Returns:
        Tuple of partisanship DataFrames, one per data source.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    raw_dir = paths.raw_dir

    partisan_paths = _resolve_paths(raw_dir, ecfg["raw_data_paths"]["partisanship"])
    _validate_paths(partisan_paths, "partisanship")

    logger.info("Loading partisanship scores datasets...")
    dfs = tuple(load_user_score_jsonl(p, max_rows_per_file) for p in partisan_paths)
    logger.success("Partisanship scores loading completed successfully!")
    return dfs


def load_demographic_data(
    election: str = DEFAULT_ELECTION,
    max_rows_per_file: Optional[int] = None,
) -> Tuple[pd.DataFrame, ...]:
    """
    Load demographic inference data for all datasets of the given election.

    Args:
        election: Election code (e.g. "us20").
        max_rows_per_file: Optional limit on rows per file.

    Returns:
        Tuple of demographic DataFrames, one per data source.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    raw_dir = paths.raw_dir

    demo_paths = _resolve_paths(raw_dir, ecfg["raw_data_paths"]["demographics"])
    _validate_paths(demo_paths, "demographic")

    logger.info("Loading demographic inference datasets...")
    dfs = tuple(load_user_demographics_jsonl(p, max_rows_per_file) for p in demo_paths)
    logger.success("Demographic inference data loading completed successfully!")
    return dfs


def load_engagement_data(
    election: str = DEFAULT_ELECTION,
    max_rows_per_file: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load retweeter and favoriter engagement data across all datasets.

    Args:
        election: Election code (e.g. "us20").
        max_rows_per_file: Optional limit on rows per file.

    Returns:
        Tuple of (retweeters_df, favoriters_df) combined DataFrames.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    raw_dir = paths.raw_dir

    retweeter_paths = _resolve_paths(raw_dir, ecfg["raw_data_paths"]["retweeters"])
    favoriter_paths = _resolve_paths(raw_dir, ecfg["raw_data_paths"]["favoriters"])
    _validate_paths(retweeter_paths, "retweeter engagement")
    _validate_paths(favoriter_paths, "favoriter engagement")

    logger.info("Loading engagement data (retweeters and favoriters)...")

    # Load and combine retweeter data
    retweeter_dfs = []
    for path in retweeter_paths:
        df = load_engagement_jsonl(path, max_rows_per_file)
        if not df.empty:
            df['data_source'] = path.parent.name
            retweeter_dfs.append(df)

    # Load and combine favoriter data
    favoriter_dfs = []
    for path in favoriter_paths:
        df = load_engagement_jsonl(path, max_rows_per_file)
        if not df.empty:
            df['data_source'] = path.parent.name
            favoriter_dfs.append(df)

    all_retweeters_df = pd.concat(retweeter_dfs, ignore_index=True) if retweeter_dfs else pd.DataFrame()
    all_favoriters_df = pd.concat(favoriter_dfs, ignore_index=True) if favoriter_dfs else pd.DataFrame()

    logger.success(f"Loaded {len(all_retweeters_df)} retweeter records and {len(all_favoriters_df)} favoriter records")

    return all_retweeters_df, all_favoriters_df


def get_base_dataset(
    election: str = DEFAULT_ELECTION,
    max_rows_per_file: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a unified poll-level dataset from raw sources. Excludes polls with 0 votes.

    Args:
        election: Election code (e.g. "us20").
        max_rows_per_file: Optional limit on rows per file.

    Returns:
        DataFrame combining polls, demographics, and audience metrics.
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    source_names = ecfg["data_source_names"]

    cache_path = paths.processed_dir / "base_dataset_cache.pkl"
    if cache_path.exists() and max_rows_per_file is None:
        logger.info(f"Loading cached base dataset from {cache_path}")
        return pd.read_pickle(cache_path)

    logger.info(f"Creating base poll dataset for [{election}]...")

    # Load raw poll data from all sources
    poll_dfs = load_polls(election, max_rows_per_file)

    # Load auxiliary datasets (partisanship, demographics, engagement)
    logger.info("Loading auxiliary data sources...")
    partisanship_dfs = load_partisanship_scores(election, max_rows_per_file)
    demographic_dfs = load_demographic_data(election, max_rows_per_file)
    retweeters_df, favoriters_df = load_engagement_data(election, max_rows_per_file)

    # Merge partisanship scores
    all_partisanship_df = pd.concat(list(partisanship_dfs), ignore_index=True)

    all_partisanship_df['user_id'] = all_partisanship_df['user_id'].astype(str)
    all_partisanship_df['score'] = pd.to_numeric(all_partisanship_df['score'], errors='coerce')
    all_partisanship_df = all_partisanship_df.drop_duplicates(subset=['user_id'], keep='first')

    # Merge demographic data
    all_demographics_df = pd.concat(list(demographic_dfs), ignore_index=True)

    all_demographics_df['user_id'] = all_demographics_df['user_id'].astype(str)
    all_demographics_df = all_demographics_df.drop_duplicates(subset=['user_id'], keep='first')

    # Extract poll metadata
    all_polls = []
    seen_poll_ids = set()

    for polls_df, source_name in zip(poll_dfs, source_names):
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

            total_votes = sum(option.get('votes', 0) for option in options)
            if total_votes == 0:
                continue

            poll_record = {
                'tweet_id': poll_id,
                'author_id': str(row.get('user', {}).get('id_str', '')),
                'created_at': row.get('created_at', ''),
                'tweet_text': row.get('text', ''),
                'data_source': source_name,
                'duration_minutes': poll.get('duration_minutes', None),
                'n_options': len(options),
                'total_votes': total_votes
            }

            poll_options = []
            for option in options:
                option_text = option.get('label', option.get('text', '')).strip()
                votes = option.get('votes', 0)
                position = option.get('position', 0)
                poll_options.append({'position': position, 'label': option_text, 'votes': votes})

            poll_record['poll_options'] = str(poll_options)
            all_polls.append(poll_record)

    unified_df = pd.DataFrame(all_polls)
    logger.info(f"Extracted {len(unified_df)} valid polls >0 votes")

    if unified_df.empty:
        return unified_df

    unified_df['tweet_id'] = unified_df['tweet_id'].astype(str)
    unified_df['author_id'] = unified_df['author_id'].astype(str)

    # Attach author characteristics
    unified_df = unified_df.merge(
        all_partisanship_df[['user_id', 'score']].rename(columns={'score': 'author_partisanship_raw'}),
        left_on='author_id', right_on='user_id', how='left'
    )

    partisanship_raw = unified_df['author_partisanship_raw']
    partisanship_mean = partisanship_raw.mean()
    partisanship_std = partisanship_raw.std()

    if pd.notna(partisanship_mean) and pd.notna(partisanship_std) and partisanship_std > 0:
        unified_df['author_partisanship'] = (partisanship_raw - partisanship_mean) / partisanship_std
    else:
        unified_df['author_partisanship'] = partisanship_raw

    unified_df = unified_df.merge(
        all_demographics_df[[
            'user_id', 'org_is_org_prob',
            'gender_male_prob',
            'age_under_29_prob', 'age_30_39_prob', 'age_40_over_prob',
        ]].rename(columns={
            'org_is_org_prob': 'author_org_prob',
            'gender_male_prob': 'author_gender_male',
            'age_under_29_prob': 'author_age_under_29',
            'age_30_39_prob': 'author_age_30_39',
            'age_40_over_prob': 'author_age_40_over',
        }),
        left_on='author_id', right_on='user_id', how='left', suffixes=('', '_demo')
    )

    # Compute audience metrics
    logger.info("Calculating audience metrics...")
    audience_engagement = defaultdict(lambda: {'users': set(), 'partisanship_scores': []})

    logger.info("Building user partisanship dictionary for fast lookups...")
    partisanship_dict = dict(zip(all_partisanship_df['user_id'].astype(str), all_partisanship_df['score']))
    valid_poll_ids = set(unified_df['tweet_id'])

    def process_interactions(interactions_df):
        if interactions_df.empty or 'tweet_id' not in interactions_df.columns:
            return
        for _, row in interactions_df.iterrows():
            tweet_id = str(row['tweet_id'])
            users_list = row.get('users_list', [])
            if isinstance(users_list, list):
                if tweet_id not in valid_poll_ids:
                    continue
                for user_obj in users_list:
                    if isinstance(user_obj, dict):
                        user_id = str(user_obj.get('id', ''))
                        if user_id and user_id not in audience_engagement[tweet_id]['users']:
                            if user_id in partisanship_dict:
                                score = partisanship_dict[user_id]
                                if pd.notna(score):
                                    audience_engagement[tweet_id]['users'].add(user_id)
                                    audience_engagement[tweet_id]['partisanship_scores'].append(score)

    logger.info("Processing retweeters...")
    process_interactions(retweeters_df)

    logger.info("Processing favoriters...")
    process_interactions(favoriters_df)

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

    if max_rows_per_file is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        unified_df.to_pickle(cache_path)
        logger.info(f"Cached base dataset to {cache_path}")

    logger.success(f"Created base dataset with {len(unified_df)} polls and {len(unified_df.columns)} features")
    return unified_df

if __name__ == "__main__":
    logger.warning("dataset.py is not intended to be run directly. Use features.py instead.")
