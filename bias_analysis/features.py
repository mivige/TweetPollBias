"""
Feature extraction for Twitter poll bias analysis.

This module transforms raw Twitter poll data into structured features for statistical analysis.
The primary research focus is understanding how candidate positioning and presentation
in Twitter polls influences voting behavior and outcomes.

Key functionalities:
- Candidate order analysis: Extracts position effects in poll options
- Appellative formality: Analyzes formal vs informal candidate references
- Smart candidate matching: Handles real-world name variations and social media conventions

The module processes data from three Twitter streams (Decahose, vote, voting) collected
during the 2020 US presidential election to enable comprehensive bias analysis.
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
    # Map of known candidate name variations found in Twitter polls
    # These patterns were identified through data exploration and capture the most common ways
    # users refer to the 2020 election candidates in poll options
    main_candidates = {
        'Trump': ['Trump', 'trump', 'TRUMP', 'Donald Trump', 'donald trump', 
                 'President Trump', 'Donald J. Trump', 'Donald J Trump'],
        'Biden': ['Biden', 'biden', 'BIDEN', 'Joe Biden', 'joe biden', 
                 'President Biden', 'Joe Biden (Democrat)', 'Sleepy Joe', 'sleepy joe'],
        'Other': []  # Catch-all for third-party candidates or unrecognized entries
    }
    
    # Build reverse lookup dictionary for O(1) exact matching performance
    candidate_mapping = {}
    for main_name, variations in main_candidates.items():
        for variation in variations:
            candidate_mapping[variation.strip()] = main_name
    
    if not candidate_text or not isinstance(candidate_text, str):
        return 'Other'
    
    # Fast path: check for exact matches first (covers ~80% of cases)
    normalized_text = candidate_text.strip()
    if normalized_text in candidate_mapping:
        return candidate_mapping[normalized_text]
    
    # Fuzzy matching path: handle emoji-decorated and prefix/suffix variations
    import re
    
    # Strip emojis and symbols while preserving core text structure
    # This handles cases like "Trump 🇺🇸" or "Biden 💙"
    cleaned_text = re.sub(r'[^\w\s\-\.\(\)\/]', ' ', normalized_text)
    
    # Remove common poll option prefixes that don't affect candidate identity
    # Examples: "Vote for Trump" → "Trump", "I choose Biden" → "Biden"
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
    
    # Remove common poll option suffixes
    # Examples: "Trump 2020" → "Trump", "Biden for president" → "Biden"
    suffixes_to_remove = [
        r'\s+(?:2020|2024|for president|for pres|presidency|administration|admin)$',
        r'\s+(?:ticket|campaign|rally|supporters?)$',
        r'\s+(?:wins?|victory|loses?|defeat)$',
        r'\s+(?:\(democrat\)|\(republican\)|\(gop\)|\(dem\)|\(r\)|\(d\))$',
        r'\s+(?:again|still|now|then|too|also)$'
    ]
    
    for suffix_pattern in suffixes_to_remove:
        cleaned_text = re.sub(suffix_pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # Normalize whitespace after text cleaning
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Check if cleaning revealed a known variation
    if cleaned_text in candidate_mapping:
        return candidate_mapping[cleaned_text]
    
    # Final fallback: regex pattern matching for partial name detection
    # This catches cases where only part of the name is recognizable
    cleaned_lower = cleaned_text.lower()
    
    # Trump name patterns: look for "trump" with optional first name variations
    trump_patterns = [
        r'\btrump\b', r'\bdonald\b.*\btrump\b', r'\btrump\b.*\bdonald\b',
        r'\bpresident\s+trump\b', r'\bdonald\s+j\.?\s+trump\b'
    ]
    
    # Biden name patterns: look for "biden" with optional first name variations
    biden_patterns = [
        r'\bbiden\b', r'\bjoe\b.*\bbiden\b', r'\bbiden\b.*\bjoe\b',
        r'\bpresident\s+biden\b', r'\bjoseph\s+biden\b'
    ]
    
    # Check patterns in order of specificity
    for pattern in trump_patterns:
        if re.search(pattern, cleaned_lower):
            return 'Trump'
    
    for pattern in biden_patterns:
        if re.search(pattern, cleaned_lower):
            return 'Biden'
    
    # No recognizable candidate found
    return 'Other'


@app.command()
def candidate_order():
    """
    Extracts candidate positioning and voting patterns from Twitter polls.
    
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
    
    # Load the three Twitter poll datasets
    # Each dataset captures different aspects of Twitter poll behavior during 2020 election
    decahose_df, vote_df, voting_df = load_for_candidate_order()
    
    # Process datasets sequentially to maintain memory efficiency
    # Each dataset is labeled to enable source-specific analysis if needed
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
                
            # Focus on first poll in tweet (multiple polls per tweet are rare)
            poll = entities['polls'][0]
            options = poll.get('options', [])
            
            if not options:
                continue
            
            # Calculate vote totals for percentage normalization
            # Some polls may have zero votes if captured immediately after posting
            total_votes = sum(option.get('votes', 0) for option in options)
            if total_votes == 0:
                total_votes = 1  # Avoid division by zero in percentage calculations
            
            # Create poll record with essential metadata
            poll_record = {
                'poll_id': str(row.get('id', '')),
                'tweet_text': row.get('text', ''),
                'dataset_source': dataset_name,
                'created_at': row.get('created_at', ''),
                'user_id': str(row.get('user', {}).get('id', '')),
                'total_votes': total_votes,
                'num_candidates': len(options)
            }
            
            # Pre-populate candidate columns with None values
            # This ensures consistent CSV structure even when candidates aren't found
            main_candidates = ['Trump', 'Biden', 'Other']
            for candidate in main_candidates:
                poll_record[f'{candidate}_position'] = None
                poll_record[f'{candidate}_votes'] = None
                poll_record[f'{candidate}_percentage'] = None
            
            # Extract data from each poll option
            for option in options:
                candidate_name = option.get('label', option.get('text', '')).strip()
                position = option.get('position', 0)
                votes = option.get('votes', 0)
                vote_percentage = (votes / total_votes) * 100 if total_votes > 0 else 0
                
                # Normalize candidate name using smart matching
                # This handles variations like "President Trump" vs "Trump" vs "Donald Trump"
                normalized_name = smart_candidate_match(candidate_name)
                
                # Store data for first occurrence of each candidate
                # Handles edge cases where a candidate appears multiple times in same poll
                if poll_record[f'{normalized_name}_position'] is None:
                    poll_record[f'{normalized_name}_position'] = position
                    poll_record[f'{normalized_name}_votes'] = votes
                    poll_record[f'{normalized_name}_percentage'] = vote_percentage
            
            candidate_order_data.append(poll_record)
    
    # Convert to pandas DataFrame for analysis and export
    result_df = pd.DataFrame(candidate_order_data)
    
    if result_df.empty:
        logger.warning("No poll data found in any dataset!")
        return
    
    # Export to CSV for analysis in other tools (R, Python notebooks, etc.)
    output_path = PROCESSED_DATA_DIR / "candidate_order_features.csv"
    result_df.to_csv(output_path, index=False)
    
    # Log summary statistics for data quality assessment
    logger.success(f"Candidate order features saved to {output_path}")
    logger.info(f"Generated {len(result_df)} poll records")
    logger.info(f"Columns created: {len(result_df.columns)}")
    
    # Report candidate coverage to assess data quality
    main_candidates = ['Trump', 'Biden', 'Other']
    for candidate in main_candidates:
        count = result_df[f'{candidate}_position'].notna().sum()
        logger.info(f"Polls with {candidate}: {count}")
    
    return result_df

@app.command()
def political_leaning():
    """
    Analyzes political leaning indicators in poll text and voting patterns.
    
    This function is designed to extract and quantify political orientation signals
    from both poll questions and voting behavior. Implementation pending based on
    research requirements and available political keyword lexicons.
    """
    logger.info("Political leaning analysis not yet implemented")
    logger.info("Planned features:")
    logger.info("- Keyword-based political orientation scoring")
    logger.info("- Partisan language detection in poll framing")
    logger.info("- Cross-reference with voting patterns")

@app.command()
def formal_vs_informal():
    """
    Extracts formal vs informal appellatives for Trump and Biden from poll options and tweet texts.
    
    Uses spaCy NER to identify person entities and appellatives, creating a dataset for manual
    labeling of formality. Each poll generates one row with appellatives found in both poll
    options and tweet text, along with vote data and suggested formality labels.
    
    Output CSV structure:
    - Trump_text: First appellative found for Trump (uncutted)
    - Trump_label: Suggested formality label (to be manually refined)
    - Trump_votes: Vote count for Trump option
    - Trump_percentage: Vote percentage for Trump option
    - Biden_text: First appellative found for Biden (uncutted)
    - Biden_label: Suggested formality label (to be manually refined)
    - Biden_votes: Vote count for Biden option  
    - Biden_percentage: Vote percentage for Biden option
    """
    logger.info("Starting formal vs informal appellatives extraction...")
    
    # Initialize spaCy NLP pipeline for named entity recognition
    # The English model provides person entity detection and linguistic analysis
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy English model successfully")
    except ImportError:
        logger.error("spaCy not installed. Please install with: pip install spacy")
        logger.error("Then download the model with: python -m spacy download en_core_web_sm")
        return
    except OSError:
        logger.error("spaCy English model not found. Please download with: python -m spacy download en_core_web_sm")
        return
    
    def suggest_formality_label(appellative_text):
        """
        Automatically categorizes appellatives by formality level.
        
        Uses pattern matching to suggest whether an appellative is formal (titles, honorifics),
        informal (nicknames, derogatory terms), or neutral (standard names). These suggestions
        provide a starting point for manual annotation and ensure consistency in labeling.
        
        Args:
            appellative_text (str): The appellative text to analyze
            
        Returns:
            str: Suggested formality label ('formal', 'informal', 'neutral')
        """
        if not appellative_text:
            return 'neutral'
        
        text_lower = appellative_text.lower().strip()
        
        # Formal appellatives: official titles, honorifics, institutional references
        # These suggest respect for office or formal political discourse
        formal_patterns = [
            'president', 'mr.', 'mr ', 'senator', 'vice president', 'former president',
            'the president', 'commander in chief', 'potus', 'democratic nominee',
            'republican nominee', 'candidate', 'former vice president'
        ]
        
        # Informal appellatives: nicknames, disparaging terms, casual references
        # These suggest familiarity, partisanship, or deliberate diminishment of authority
        informal_patterns = [
            'sleepy', 'crooked', 'crazy', 'lyin', 'little', 'corrupt', 'basement',
            'dementia', 'orange', 'donnie', 'joey', 'joe', 'don', 'creepy'
        ]
        
        # Check for formal indicators first (tend to be more specific)
        for pattern in formal_patterns:
            if pattern in text_lower:
                return 'formal'
        
        # Then check for informal indicators
        for pattern in informal_patterns:
            if pattern in text_lower:
                return 'informal'
        
        # Default classification for standard name usage without clear formality markers
        return 'neutral'
    
    def extract_appellatives_from_text(text, target_candidates=['trump', 'biden']):
        """
        Identifies and extracts candidate appellatives from text using NLP.
        
        Combines spaCy's named entity recognition with regex pattern matching to find
        references to target candidates. Captures surrounding context to preserve
        titles, descriptors, and modifiers that indicate formality level.
        
        Args:
            text (str): Text to analyze (poll option or tweet content)
            target_candidates (list): Candidates to search for
            
        Returns:
            dict: Mapping of candidate names to first appellative found
        """
        if not text or not isinstance(text, str):
            return {}
        
        appellatives = {}
        
        # Use spaCy NER to identify person entities and their context
        doc = nlp(text)
        
        # Extract person entities with surrounding context for complete appellatives
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Expand context to capture titles and descriptors
                # Include 3 tokens before and after the entity to get full appellative
                start_idx = max(0, ent.start - 3)
                end_idx = min(len(doc), ent.end + 3)
                
                full_span = doc[start_idx:end_idx].text
                entity_text = ent.text.lower()
                
                # Check if this entity refers to our target candidates
                for candidate in target_candidates:
                    if candidate in entity_text and candidate not in appellatives:
                        # Clean up extracted appellative by removing non-essential words
                        appellative = full_span.strip()
                        import re
                        # Remove conjunctions and prepositions that don't add meaning
                        appellative = re.sub(r'^(and|or|but|the|a|an|with|for|against|vs|versus)\s+', '', appellative, flags=re.IGNORECASE)
                        appellative = re.sub(r'\s+(and|or|but|with|for|against|vs|versus|will|would|should|could)$', '', appellative, flags=re.IGNORECASE)
                        appellatives[candidate] = appellative.strip()
        
        # Fallback: regex pattern matching for cases where NER misses candidates
        # This is essential because spaCy may not recognize informal appellatives or nicknames
        for candidate in target_candidates:
            if candidate not in appellatives:
                import re
                patterns = []
                
                if candidate == 'trump':
                    patterns = [
                        r'\b(?:president\s+)?(?:donald\s+)?(?:j\.?\s+)?trump\b',
                        r'\btrump\b',
                        r'\b(?:mr\.?\s+)?trump\b',
                        r'\b(?:sleepy|crooked|crazy|orange|donnie)\b.*?\btrump\b',
                        r'\btrump.*?\b(?:donald|don|donnie)\b'
                    ]
                elif candidate == 'biden':
                    patterns = [
                        r'\b(?:president\s+)?(?:joe\s+)?(?:joseph\s+)?biden\b',
                        r'\bbiden\b',
                        r'\b(?:mr\.?\s+)?biden\b',
                        r'\b(?:sleepy|creepy|basement|dementia)\s+(?:joe\s+)?biden\b',
                        r'\bbiden.*?\b(?:joe|joey|joseph)\b'
                    ]
                
                # Apply patterns and capture first match
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        appellatives[candidate] = match.group(0).strip()
                        break
        
        return appellatives
    
    # Load poll data from the same sources as candidate order analysis
    # Ensures consistency across different feature extraction processes
    decahose_df, vote_df, voting_df = load_for_candidate_order()
    
    # Process each dataset sequentially for memory efficiency
    all_dataframes = [
        (decahose_df, "decahose"),
        (vote_df, "vote"), 
        (voting_df, "voting")
    ]
    
    appellative_data = []
    
    for df, dataset_name in all_dataframes:
        logger.info(f"Processing {dataset_name} dataset with {len(df)} records for appellatives...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting appellatives from {dataset_name}"):
            # Validate Twitter poll structure
            if not isinstance(row.get('entities'), dict):
                continue
                
            entities = row['entities']
            if 'polls' not in entities or not entities['polls']:
                continue
                
            # Focus on first poll in tweet
            poll = entities['polls'][0]
            options = poll.get('options', [])
            
            if not options:
                continue
            
            # Calculate vote statistics for context
            total_votes = sum(option.get('votes', 0) for option in options)
            if total_votes == 0:
                total_votes = 1
            
            # Initialize record with minimal metadata
            poll_record = {
                'poll_id': str(row.get('id', '')),
                'total_votes': total_votes,
                'num_options': len(options)
            }
            
            # Pre-populate appellative columns for consistent CSV structure
            for candidate in ['Trump', 'Biden']:
                poll_record[f'{candidate}_text'] = None      # Raw appellative found
                poll_record[f'{candidate}_label'] = None     # Suggested formality label
                poll_record[f'{candidate}_votes'] = None     # Vote count for this candidate
                poll_record[f'{candidate}_percentage'] = None # Vote percentage for this candidate
            
            # Extract appellatives from main tweet text first
            # This captures how the poll author refers to candidates in context
            tweet_appellatives = extract_appellatives_from_text(row.get('text', ''))
            
            # Process each poll option to find candidate appellatives and vote data
            candidate_votes = {}
            for option in options:
                option_text = option.get('label', option.get('text', '')).strip()
                votes = option.get('votes', 0)
                vote_percentage = (votes / total_votes) * 100 if total_votes > 0 else 0
                
                # Extract appellatives from this specific poll option
                option_appellatives = extract_appellatives_from_text(option_text)
                
                # Determine which candidate this option represents
                normalized_candidate = smart_candidate_match(option_text)
                
                if normalized_candidate in ['Trump', 'Biden']:
                    # Store vote data for this candidate
                    candidate_votes[normalized_candidate] = {
                        'votes': votes,
                        'percentage': vote_percentage
                    }
                    
                    # Select best appellative: prioritize poll option over tweet text
                    # Poll options are more direct references to candidates
                    appellative_text = None
                    candidate_lower = normalized_candidate.lower()
                    
                    if candidate_lower in option_appellatives:
                        appellative_text = option_appellatives[candidate_lower]
                    elif candidate_lower in tweet_appellatives:
                        appellative_text = tweet_appellatives[candidate_lower]
                    
                    # Store first appellative found for this candidate in this poll
                    if appellative_text and poll_record[f'{normalized_candidate}_text'] is None:
                        poll_record[f'{normalized_candidate}_text'] = appellative_text
                        poll_record[f'{normalized_candidate}_label'] = suggest_formality_label(appellative_text)
            
            # Add vote statistics for all candidates found
            for candidate, vote_data in candidate_votes.items():
                poll_record[f'{candidate}_votes'] = vote_data['votes']
                poll_record[f'{candidate}_percentage'] = vote_data['percentage']
            
            appellative_data.append(poll_record)
    
    # Convert to DataFrame for export and analysis
    result_df = pd.DataFrame(appellative_data)
    
    if result_df.empty:
        logger.warning("No appellative data found in any dataset!")
        return
    
    # Export streamlined CSV for manual annotation workflow
    output_path = PROCESSED_DATA_DIR / "formal_informal_appellatives.csv"
    result_df.to_csv(output_path, index=False)
    
    # Report extraction statistics
    logger.success(f"Formal vs informal appellatives saved to {output_path}")
    logger.info(f"Generated {len(result_df)} poll records with appellatives")
    
    # Show coverage and initial formality distribution
    for candidate in ['Trump', 'Biden']:
        text_count = result_df[f'{candidate}_text'].notna().sum()
        votes_count = result_df[f'{candidate}_votes'].notna().sum()
        logger.info(f"Polls with {candidate} appellatives: {text_count}")
        logger.info(f"Polls with {candidate} votes: {votes_count}")
        
        if text_count > 0:
            # Display automatic formality classification results
            labels = result_df[f'{candidate}_label'].value_counts()
            logger.info(f"{candidate} formality distribution: {dict(labels)}")
    
    return result_df

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
