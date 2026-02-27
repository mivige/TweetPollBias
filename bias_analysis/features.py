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
import numpy as np
import json
import typing
from collections import defaultdict

from loguru import logger
from tqdm import tqdm
import typer

from bias_analysis.config import PROCESSED_DATA_DIR
from bias_analysis.dataset import get_base_dataset
import ast

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
    
    # Load the unified foundational dataset
    base_df = get_base_dataset()
    
    if base_df.empty:
        logger.warning("Base dataset is empty. Cannot extract candidate order.")
        return
        
    candidate_order_data = []
    
    logger.info(f"Processing candidate order for {len(base_df)} polls...")
        
    for idx, row in tqdm(base_df.iterrows(), total=len(base_df), desc="Extracting candidate order"):
        
        try:
            options = ast.literal_eval(row['poll_options'])
        except Exception as e:
            continue
            
        if not options:
            continue
        
        total_votes = row['total_votes']
        
        # Create poll record with essential metadata
        poll_record = {
            'poll_id': str(row['tweet_id']),
            'tweet_text': row['tweet_text'],
            'dataset_source': row['data_source'],
            'created_at': row['created_at'],
            'user_id': str(row['author_id']),
            'total_votes': total_votes,
            'num_candidates': row['n_options']
        }
        
        # Pre-populate candidate columns with None values
        main_candidates = ['Trump', 'Biden', 'Other']
        for candidate in main_candidates:
            poll_record[f'{candidate}_position'] = None
            poll_record[f'{candidate}_votes'] = None
            poll_record[f'{candidate}_percentage'] = None
        
        # Extract data from each poll option
        for option in options:
            candidate_name = option.get('label', '')
            position = option.get('position', 0)
            votes = option.get('votes', 0)
            vote_percentage = (votes / total_votes) * 100 if total_votes > 0 else 0
            
            normalized_name = smart_candidate_match(candidate_name)
            
            if poll_record[f'{normalized_name}_position'] is None:
                poll_record[f'{normalized_name}_position'] = position
                poll_record[f'{normalized_name}_votes'] = votes
                poll_record[f'{normalized_name}_percentage'] = vote_percentage
            else:
                poll_record[f'{normalized_name}_votes'] = int(poll_record[f'{normalized_name}_votes'] or 0) + int(votes)
                poll_record[f'{normalized_name}_percentage'] = float(poll_record[f'{normalized_name}_percentage'] or 0.0) + float(vote_percentage)
        
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
def political_leaning(max_samples: int = typer.Option(10, help="Maximum number of samples to process from each dataset")):
    """
    Analyzes political leaning indicators in poll text and voting patterns.
    
    Uses the facebook/bart-large-mnli model from Hugging Face to classify
    political orientation of Twitter poll texts using natural language inference.
    The model evaluates six political hypotheses against each poll's tweet text:
    - Conservative vs liberal policy support
    - Support for Donald Trump vs Joe Biden  
    - Opposition to Donald Trump vs Joe Biden
    
    Output CSV structure:
    - conservative_score: Conservative policy support probability (0.0-1.0)
    - liberal_score: Liberal policy support probability (0.0-1.0)
    - trump_support_score: Trump support probability (0.0-1.0)
    - biden_support_score: Biden support probability (0.0-1.0)
    - anti_trump_score: Trump opposition probability (0.0-1.0)
    - anti_biden_score: Biden opposition probability (0.0-1.0)
    - Trump_votes: Vote count for Trump option in poll
    - Trump_percentage: Vote percentage for Trump option in poll
    - Biden_votes: Vote count for Biden option in poll
    - Biden_percentage: Vote percentage for Biden option in poll
    
    Args:
        max_samples: Limit processing to this many samples per dataset for testing
    """
    logger.info("Starting political leaning analysis using facebook/bart-large-mnli...")
    
    # Initialize the BART-MNLI model and tokenizer for natural language inference
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        logger.info("Loading facebook/bart-large-mnli model...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

        # Set model to evaluation mode to disable dropout and batch normalization training behavior
        model.eval()
        logger.success("facebook/bart-large-mnli model loaded successfully")

    except ImportError:
        logger.error("transformers library not installed. Please install with: pip install transformers torch")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Define political hypotheses optimized for BART-MNLI natural language inference
    # These hypotheses use direct text analysis format that works best with BART-MNLI
    # Each hypothesis will be evaluated against tweet text to generate probability scores
    hypotheses = [
        "The text supports conservative politics",
        "The text supports liberal politics", 
        "The text praises Donald Trump",
        "The text praises Joe Biden",
        "The text criticizes Donald Trump",
        "The text criticizes Joe Biden"
    ]
    
    def classify_political_leaning(text, hypotheses):
        """
        Classify political leaning of text using the facebook/bart-large-mnli model.
        
        For each hypothesis, the model determines if the text entails (supports) 
        the hypothesis using natural language inference. Returns probability scores
        where higher values indicate stronger evidence for each political orientation.
        
        Args:
            text (str): Tweet text to analyze for political content
            hypotheses (list): Political hypotheses to evaluate against the text
            
        Returns:
            dict: Mapping of hypothesis to entailment probability (0.0-1.0)
        """
        if not text or not isinstance(text, str):
            return {hyp: 0.0 for hyp in hypotheses}
        
        results = {}
        
        # Evaluate each political hypothesis against the tweet text using NLI
        for hypothesis in hypotheses:
            try:
                # Tokenize the text-hypothesis pair for BART-MNLI processing
                # Model expects premise (text) and hypothesis as separate inputs
                inputs = tokenizer(
                    text, 
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Run inference without gradient computation for efficiency
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    # Extract entailment probability from BART-MNLI output logits
                    # BART-MNLI returns [contradiction, neutral, entailment] probabilities
                    # Index 2 corresponds to entailment (text supports hypothesis)
                    entailment_score = predictions[0][2].item()
                    results[hypothesis] = entailment_score
                    
            except Exception as e:
                logger.warning(f"Error processing hypothesis '{hypothesis}' for text: {e}")
                results[hypothesis] = 0.0
        
        return results
    
    # Load the unified foundational dataset
    base_df = get_base_dataset()
    
    if base_df.empty:
        logger.warning("Base dataset is empty. Cannot extract political leaning.")
        return
        
    sample_df = base_df.head(max_samples) if len(base_df) > max_samples else base_df
    logger.info(f"Processing political leaning for {len(sample_df)} polls (limited to {max_samples} samples)...")
    
    political_leaning_data = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analyzing political leaning"):
        try:
            options = ast.literal_eval(row['poll_options'])
        except Exception:
            continue
            
        if not options:
            continue
            
        tweet_text = row.get('tweet_text', '').strip()
        if not tweet_text:
            continue
            
        # Classify political orientation of the tweet text using BART-MNLI
        political_scores = classify_political_leaning(tweet_text, hypotheses)
        
        total_votes = row['total_votes']
        if total_votes == 0:
            total_votes = 1  # Prevent division by zero
            
        # Initialize poll record with essential metadata
        poll_record = {
            'poll_id': str(row['tweet_id']),
            'total_votes': total_votes,
            'num_options': row['n_options']
        }
        
        # Initialize candidate vote columns to ensure consistent CSV structure
        for candidate in ['Trump', 'Biden']:
            poll_record[f'{candidate}_votes'] = None
            poll_record[f'{candidate}_percentage'] = None
            
        # Store political leaning scores from BART-MNLI classification
        poll_record['conservative_score'] = political_scores.get("The text supports conservative politics", 0.0)
        poll_record['liberal_score'] = political_scores.get("The text supports liberal politics", 0.0)
        poll_record['trump_support_score'] = political_scores.get("The text praises Donald Trump", 0.0)
        poll_record['biden_support_score'] = political_scores.get("The text praises Joe Biden", 0.0)
        poll_record['anti_trump_score'] = political_scores.get("The text criticizes Donald Trump", 0.0)
        poll_record['anti_biden_score'] = political_scores.get("The text criticizes Joe Biden", 0.0)
        
        # Extract vote data from poll options to correlate with political leaning
        candidate_votes = {}
        for option in options:
            option_text = option.get('label', '')
            votes = option.get('votes', 0)
            vote_percentage = (votes / total_votes) * 100 if total_votes > 0 else 0
            
            normalized_candidate = smart_candidate_match(option_text)
            
            if normalized_candidate in ['Trump', 'Biden']:
                if normalized_candidate not in candidate_votes:
                    candidate_votes[normalized_candidate] = {
                        'votes': votes,
                        'percentage': vote_percentage
                    }
                else:
                    candidate_votes[normalized_candidate]['votes'] += votes
                    candidate_votes[normalized_candidate]['percentage'] += vote_percentage
                
        # Populate candidate voting data in the poll record
        for candidate, vote_data in candidate_votes.items():
            poll_record[f'{candidate}_votes'] = vote_data['votes']
            poll_record[f'{candidate}_percentage'] = vote_data['percentage']
            
        political_leaning_data.append(poll_record)
    
    # Convert collected data to DataFrame for analysis and export
    result_df = pd.DataFrame(political_leaning_data)
    
    if result_df.empty:
        logger.warning("No political leaning data generated!")
        return
    
    # Export results to CSV for downstream analysis and visualization
    output_path = PROCESSED_DATA_DIR / "political_leaning_features.csv"
    result_df.to_csv(output_path, index=False)
    
    # Report analysis completion and summary statistics
    logger.success("Political leaning analysis completed!")
    logger.success(f"Political leaning features saved to {output_path}")
    logger.info(f"Generated {len(result_df)} poll records with political leaning scores")
    logger.info(f"Columns created: {len(result_df.columns)}")
    
    # Display statistical summary of political leaning classification results
    logger.info("Political leaning score statistics:")
    
    # Report mean and standard deviation for each political dimension
    score_cols = ['conservative_score', 'liberal_score', 'trump_support_score', 'biden_support_score', 'anti_trump_score', 'anti_biden_score']
    for score_col in score_cols:
        if score_col in result_df.columns:
            mean_score = result_df[score_col].mean()
            std_score = result_df[score_col].std()
            logger.info(f"  {score_col}: mean={mean_score:.3f}, std={std_score:.3f}")
    
    # Report data coverage for correlation analysis
    for candidate in ['Trump', 'Biden']:
        votes_count = result_df[f'{candidate}_votes'].notna().sum()
        logger.info(f"Polls with {candidate} votes: {votes_count}")
    
    return result_df

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
    
    # Load the unified foundational dataset
    base_df = get_base_dataset()
    
    if base_df.empty:
        logger.warning("Base dataset is empty. Cannot extract appellatives.")
        return
        
    appellative_data = []
    
    logger.info(f"Processing appellatives for {len(base_df)} polls...")
        
    for idx, row in tqdm(base_df.iterrows(), total=len(base_df), desc="Extracting appellatives"):
        try:
            options = ast.literal_eval(row['poll_options'])
        except Exception:
            continue
            
        if not options:
            continue
            
        total_votes = row['total_votes']
        if total_votes == 0:
            total_votes = 1
            
        # Initialize record with minimal metadata
        poll_record = {
            'poll_id': str(row['tweet_id']),
            'total_votes': total_votes,
            'num_options': row['n_options']
        }
        
        # Pre-populate appellative columns for consistent CSV structure
        for candidate in ['Trump', 'Biden']:
            poll_record[f'{candidate}_text'] = None      # Raw appellative found
            poll_record[f'{candidate}_label'] = None     # Suggested formality label
            poll_record[f'{candidate}_votes'] = None     # Vote count for this candidate
            poll_record[f'{candidate}_percentage'] = None # Vote percentage for this candidate
            
        # Extract appellatives from main tweet text
        tweet_appellatives = extract_appellatives_from_text(row.get('tweet_text', ''))
        
        # Process each poll option to find candidate appellatives and vote data
        candidate_votes = {}
        for option in options:
            option_text = option.get('label', '')
            votes = option.get('votes', 0)
            vote_percentage = (votes / total_votes) * 100 if total_votes > 0 else 0
            
            option_appellatives = extract_appellatives_from_text(option_text)
            normalized_candidate = smart_candidate_match(option_text)
            
            if normalized_candidate in ['Trump', 'Biden']:
                if normalized_candidate not in candidate_votes:
                    candidate_votes[normalized_candidate] = {
                        'votes': votes,
                        'percentage': vote_percentage
                    }
                else:
                    candidate_votes[normalized_candidate]['votes'] += votes
                    candidate_votes[normalized_candidate]['percentage'] += vote_percentage
                
                appellative_text = None
                candidate_lower = normalized_candidate.lower()
                
                if candidate_lower in option_appellatives:
                    appellative_text = option_appellatives[candidate_lower]
                elif candidate_lower in tweet_appellatives:
                    appellative_text = tweet_appellatives[candidate_lower]
                
                if appellative_text and poll_record[f'{normalized_candidate}_text'] is None:
                    poll_record[f'{normalized_candidate}_text'] = appellative_text
                    poll_record[f'{normalized_candidate}_label'] = suggest_formality_label(appellative_text)
        
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
def pearson_correlation(save_plots: bool = typer.Option(True, help="Save diagnostic plots to files"),
                       max_rows: int = typer.Option(None, help="Limit dataset size for testing")):
    """
    Computes Pearson correlations between bias markers and bias outcomes with statistical controls.
    
    Analysis workflow:
    1. Load and prepare unified poll dataset
    2. Create normalized bias measures (-1=Biden/Liberal, +1=Trump/Conservative)
    3. Check statistical assumptions (outliers, normality)
    4. Compute pairwise correlations between 3 markers and 3 outcomes
    5. Apply Benjamini-Hochberg FDR correction for multiple testing
    6. Calculate bootstrap confidence intervals for significant correlations
    7. Generate correlation heatmap and save results
    
    Args:
        save_plots: Whether to save diagnostic plots to processed data directory
        max_rows: Optional limit on dataset size for development/testing
    """
    logger.info("Starting comprehensive Pearson correlation analysis...")
    
    # Import required statistical libraries
    try:
        import scipy.stats as stats
        import statsmodels.api as sm
        import statsmodels.stats.multitest as multi
        from statsmodels.stats.contingency_tables import mcnemar
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.utils import resample
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        
    except ImportError as e:
        logger.error(f"Required statistical libraries not installed: {e}")
        logger.error("Please install with: pip install scipy statsmodels matplotlib seaborn scikit-learn")
        return
    
    # Load and validate dataset
    logger.info("Loading unified poll dataset...")
    
    from bias_analysis.dataset import create_unified_poll_dataset
    
    try:
        base_df = get_base_dataset(max_rows_per_file=max_rows)
        
        # Load marker CSVs dynamically
        markers_dir = PROCESSED_DATA_DIR
        candidate_order_df = pd.read_csv(markers_dir / "candidate_order_features.csv") if (markers_dir / "candidate_order_features.csv").exists() else pd.DataFrame()
        appellatives_df = pd.read_csv(markers_dir / "formal_informal_appellatives.csv") if (markers_dir / "formal_informal_appellatives.csv").exists() else pd.DataFrame()
        leaning_df = pd.read_csv(markers_dir / "political_leaning_features.csv") if (markers_dir / "political_leaning_features.csv").exists() else pd.DataFrame()
        
        # Merge markers into base_df
        unified_df = base_df.copy()
        
        if not candidate_order_df.empty:
            candidate_order_df['poll_id'] = candidate_order_df['poll_id'].astype(str)
            unified_df = unified_df.merge(candidate_order_df[['poll_id', 'Trump_position', 'Biden_position', 'Trump_percentage', 'Biden_percentage']], 
                                          left_on='tweet_id', right_on='poll_id', how='left')
            # Compute candidate_order bias
            # Calculate proportional position difference:
            # e.g., Biden at 1, Trump at 4 -> (1 - 4) * 0.25 = -0.75 (Favors Biden heavily)
            unified_df['candidate_order'] = (unified_df['Biden_position'] - unified_df['Trump_position']) * 0.25
            # NaN where missing both
            missing_pos_mask = unified_df['Biden_position'].isna() | unified_df['Trump_position'].isna()
            unified_df.loc[missing_pos_mask, 'candidate_order'] = np.nan
        else:
            unified_df['candidate_order'] = np.nan
            unified_df['Trump_percentage'] = np.nan
            unified_df['Biden_percentage'] = np.nan
            
        if not appellatives_df.empty:
            appellatives_df['poll_id'] = appellatives_df['poll_id'].astype(str)
            unified_df = unified_df.merge(appellatives_df[['poll_id', 'Trump_label', 'Biden_label']], 
                                          left_on='tweet_id', right_on='poll_id', how='left')
            label_map = {'formal': 1.0, 'informal': -1.0, 'neutral': 0.0}
            unified_df['trump_formal_appellative'] = unified_df['Trump_label'].map(label_map)
            unified_df['biden_formal_appellative'] = unified_df['Biden_label'].map(label_map)
        else:
            unified_df['trump_formal_appellative'] = np.nan
            unified_df['biden_formal_appellative'] = np.nan
            
        if not leaning_df.empty:
            leaning_df['poll_id'] = leaning_df['poll_id'].astype(str)
            unified_df = unified_df.merge(leaning_df[['poll_id', 'conservative_score', 'liberal_score', 'trump_support_score', 'biden_support_score', 'anti_trump_score', 'anti_biden_score']], 
                                          left_on='tweet_id', right_on='poll_id', how='left')
        else:
            for col in ['conservative_score', 'liberal_score', 'trump_support_score', 'biden_support_score', 'anti_trump_score', 'anti_biden_score']:
                unified_df[col] = np.nan
                
        # Fill poll_outcome needs (trump_share, biden_share normally expected, fallback to percentage/100)
        unified_df['trump_share'] = unified_df['Trump_percentage'] / 100.0
        unified_df['biden_share'] = unified_df['Biden_percentage'] / 100.0
    except Exception as e:
        logger.error(f"Failed to load and merge unified dataset: {e}")
        return
        
    if unified_df.empty:
        logger.error("Unified dataset is empty!")
        return
        
    logger.success(f"Merged {len(unified_df)} polls with {len(unified_df.columns)} features")
    
    # Create author bias composite score
    logger.info("Computing author bias composite score (-1=left/Biden, +1=right/Trump)...")
    
    def compute_author_bias(row):
        """
        Compute composite author bias from multiple indicators.
        
        Combines partisanship scores, candidate order, formality, and sentiment
        into a single bias measure normalized to [-1, +1] scale.
        
        Returns:
            float: -1 (Biden/Liberal) to +1 (Trump/Conservative)
        """
        bias_components = []
        
        # Author partisanship (normalized using tanh)
        if pd.notna(row.get('author_partisanship')):
            # Normalize z-score to [-1, +1] range using tanh function
            # tanh naturally maps (-∞,+∞) to (-1,+1) with most values in [-3,+3] mapping to ~[-0.99,+0.99]
            partisanship_normalized = np.tanh(row['author_partisanship'])
            bias_components.append(partisanship_normalized)
        
        # Candidate order bias (already scaled -1 to +1)
        if pd.notna(row.get('candidate_order')):
            bias_components.append(row['candidate_order'])
        
        # Formal appellatives
        if pd.notna(row.get('trump_formal_appellative')):
            bias_components.append(row['trump_formal_appellative'])
        
        if pd.notna(row.get('biden_formal_appellative')):
            bias_components.append(-row['biden_formal_appellative'])
        
        # Conservative sentiment scores (pro-Trump)
        conservative_scores = []
        for col in ['conservative_score', 'trump_support_score', 'anti_biden_score']:
            if pd.notna(row.get(col)):
                conservative_scores.append(row[col])
        
        if conservative_scores:
            # Average conservative scores and add to bias
            bias_components.append(np.mean(conservative_scores))
        
        # Liberal sentiment scores (pro-Biden, negative weight)
        liberal_scores = []
        for col in ['liberal_score', 'biden_support_score', 'anti_trump_score']:
            if pd.notna(row.get(col)):
                liberal_scores.append(row[col])
        
        if liberal_scores:
            bias_components.append(-np.mean(liberal_scores))
        
        # Average all components and clip to [-1, +1] range
        if bias_components:
            raw_bias = np.mean(bias_components)
            return np.clip(raw_bias, -1.0, 1.0)
        else:
            return np.nan
    
    # Apply bias computation to all rows
    unified_df['author_bias'] = unified_df.apply(compute_author_bias, axis=1)
    
    # Create formality bias measure
    logger.info("Computing formality bias...")
    
    def compute_formality_bias(row):
        """
        Compute formality bias based on formal appellative usage.
        
        Returns:
            -1: Only Biden formal, 0: Both/neither formal, +1: Only Trump formal
        """
        trump_formal = row.get('trump_formal_appellative', 0)
        biden_formal = row.get('biden_formal_appellative', 0)
        
        if pd.isna(trump_formal):
            trump_formal = 0
        if pd.isna(biden_formal):
            biden_formal = 0
            
        trump_formal = int(trump_formal)
        biden_formal = int(biden_formal)
        
        if trump_formal == 1 and biden_formal == 0:
            return 1.0
        elif trump_formal == 0 and biden_formal == 1:
            return -1.0
        else:
            return 0.0
    
    unified_df['formality_bias'] = unified_df.apply(compute_formality_bias, axis=1)
    
    # Log formality bias results
    formality_stats = unified_df['formality_bias'].describe()
    n_formality = unified_df['formality_bias'].notna().sum()
    
    logger.success(f"Formality bias computed for {n_formality}/{len(unified_df)} polls")
    logger.info(f"  Range: [{formality_stats['min']:.1f}, {formality_stats['max']:.1f}]")
    
    # Show formality distribution
    trump_only_formal = (unified_df['formality_bias'] == 1.0).sum()
    biden_only_formal = (unified_df['formality_bias'] == -1.0).sum()
    both_or_neither = (unified_df['formality_bias'] == 0.0).sum()
    
    logger.info(f"Formality distribution: {biden_only_formal} Biden-only formal, {both_or_neither} both/neither, {trump_only_formal} Trump-only formal")
    
    # Create audience bias and poll outcome measures
    logger.info("Computing audience bias and poll outcome measures...")
    
    def normalize_audience_bias(partisanship_score):
        """Normalize audience partisanship using tanh to [-1, +1] scale."""
        if pd.notna(partisanship_score):
            return np.tanh(partisanship_score)
        return np.nan
    
    unified_df['audience_bias'] = unified_df['audience_mean_partisanship'].apply(normalize_audience_bias)
    
    def compute_poll_outcome_bias(row):
        """Compute poll outcome as Trump advantage minus Biden advantage."""
        trump_share = row.get('trump_share')
        biden_share = row.get('biden_share')
        
        if pd.notna(trump_share) and pd.notna(biden_share):
            # Calculate Trump advantage over Biden
            # trump_share - biden_share ranges from -1 to +1 naturally
            poll_bias = trump_share - biden_share
            return np.clip(poll_bias, -1.0, 1.0)
        return np.nan
    
    unified_df['poll_outcome'] = unified_df.apply(compute_poll_outcome_bias, axis=1)
    
    # Log bias computation results
    bias_stats = unified_df['author_bias'].describe()
    audience_stats = unified_df['audience_bias'].describe() 
    outcome_stats = unified_df['poll_outcome'].describe()
    
    n_with_author = unified_df['author_bias'].notna().sum()
    n_with_audience = unified_df['audience_bias'].notna().sum()
    n_with_outcome = unified_df['poll_outcome'].notna().sum()
    
    logger.success(f"Author bias computed for {n_with_author}/{len(unified_df)} polls")
    logger.info(f"  Range: [{bias_stats['min']:.3f}, {bias_stats['max']:.3f}], Mean: {bias_stats['mean']:.3f}")
    
    logger.success(f"Audience bias computed for {n_with_audience}/{len(unified_df)} polls")
    logger.info(f"  Range: [{audience_stats['min']:.3f}, {audience_stats['max']:.3f}], Mean: {audience_stats['mean']:.3f}")
    
    logger.success(f"Poll outcome computed for {n_with_outcome}/{len(unified_df)} polls")
    logger.info(f"  Range: [{outcome_stats['min']:.3f}, {outcome_stats['max']:.3f}], Mean: {outcome_stats['mean']:.3f}")
    
    # Show bias distribution across all measures
    logger.info("\nBias distribution (-1=Biden/liberal, +1=Trump/conservative):")
    for bias_name, bias_col in [('Author', 'author_bias'), ('Audience', 'audience_bias'), ('Poll Outcome', 'poll_outcome')]:
        if bias_col in unified_df.columns:
            data = unified_df[bias_col].dropna()
            if len(data) > 0:
                left = (data < -0.1).sum()
                neutral = ((data >= -0.1) & (data <= 0.1)).sum()
                right = (data > 0.1).sum()
                logger.info(f"  {bias_name}: {left} left, {neutral} neutral, {right} right")
    
    # Define key variables for focused correlation analysis
    # All variables normalized to -1 (Biden/liberal) to +1 (Trump/conservative) scale
    key_variables = {
        'author_bias': 'author_bias',        # Composite author bias
        'audience_bias': 'audience_bias',    # Normalized audience partisanship 
        'poll_outcome': 'poll_outcome',      # Trump vs Biden poll advantage
    }
    
    # Bias markers to analyze
    bias_markers = {
        'candidate_order': 'candidate_order',          # M1: Position-based order effect (-1=Biden first, +1=Trump first, 0.25 per position)
        'formality_bias': 'formality_bias',            # M2: Formality bias (-1=Biden formal, 0=both/neither, +1=Trump formal)
        'political_leaning': 'conservative_score',     # M3: Used conservative_score directly for text partisan
    }
    
    # Validate that key variables exist in the dataset
    missing_vars = []
    for var_name, col_name in {**key_variables, **bias_markers}.items():
        if col_name not in unified_df.columns:
            missing_vars.append(f"{var_name} ({col_name})")
    
    if missing_vars:
        logger.error(f"Missing required variables: {missing_vars}")
        return
        
    logger.success("All required variables found in dataset")
    
    # Check statistical assumptions
    logger.info("Checking Pearson correlation assumptions...")
    
    def check_assumptions(df, variables_dict, save_plots=True):
        """Check outliers and normality assumptions, create distribution plots."""
        
        assumption_results: typing.Dict[str, typing.Any] = {}
        
        # Create analysis subset with complete cases only
        analysis_vars = list(variables_dict.values())
        clean_df = df[analysis_vars].dropna()
        
        if len(clean_df) < 30:
            logger.warning(f"Small sample size: {len(clean_df)} complete cases")
        else:
            logger.info(f"Analysis sample: {len(clean_df)} complete cases from {len(df)} total polls")
        
        # Check for extreme outliers (|z-score| > 3)
        outlier_counts = {}
        for var_name, col_name in variables_dict.items():
            if clean_df[col_name].dtype in ['float64', 'int64']:
                z_scores = np.abs(stats.zscore(clean_df[col_name].dropna()))
                outliers = (z_scores > 3).sum()
                outlier_counts[var_name] = outliers
                
        total_outliers = sum(outlier_counts.values())
        assumption_results['outliers'] = outlier_counts
        
        if total_outliers > 0:
            logger.warning(f"Found {total_outliers} extreme outliers (|z|>3) across variables")
            for var, count in outlier_counts.items():
                if count > 0:
                    logger.warning(f"  {var}: {count} outliers")
        else:
            logger.success("No extreme outliers detected")
            
        # Test normality for key continuous variables using Shapiro-Wilk
        normality_results = {}
        continuous_vars = ['author_bias', 'audience_bias', 'poll_outcome']
        
        for var_name in continuous_vars:
            if var_name in variables_dict:
                col_name = variables_dict[var_name]
                data = clean_df[col_name].dropna()
                
                if len(data) >= 3:  # Minimum for Shapiro-Wilk
                    # Use subsample for large datasets (Shapiro-Wilk has sample size limits)
                    test_data = data.sample(min(5000, len(data)), random_state=42) if len(data) > 5000 else data
                    stat, p_value = stats.shapiro(test_data)
                    normality_results[var_name] = {'statistic': stat, 'p_value': p_value, 'normal': p_value > 0.05}
                    
        assumption_results['normality'] = normality_results
        
        # Log normality test results
        normal_vars = [var for var, result in normality_results.items() if result['normal']]
        non_normal_vars = [var for var, result in normality_results.items() if not result['normal']]
        
        if normal_vars:
            logger.info(f"Variables with normal distribution (p>0.05): {normal_vars}")
        if non_normal_vars:
            logger.warning(f"Variables with non-normal distribution (p≤0.05): {non_normal_vars}")
            logger.info("Non-normality may affect correlation interpretation but Pearson r is robust with large samples")
        
        # Create diagnostic plots if requested
        if save_plots and len(clean_df) > 10:
            logger.info("Creating diagnostic plots...")
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create distribution plots for key bias variables
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Distribution of Bias Measures (-1=Biden/Liberal, +1=Trump/Conservative)', fontsize=14, fontweight='bold')
            
            key_vars_for_plots = ['author_bias', 'audience_bias', 'poll_outcome']
            for i, var_name in enumerate(key_vars_for_plots):
                if var_name in variables_dict:
                    ax = axes[i]
                    col_name = variables_dict[var_name]
                    data = clean_df[col_name].dropna()
                    
                    # Filter out infinite values that cause plotting issues
                    data = data[np.isfinite(data)]
                    
                    if len(data) == 0:
                        ax.text(0.5, 0.5, f'No finite data for {var_name}', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{var_name.replace("_", " ").title()}')
                        continue
                    
                    # Histogram with normal curve overlay
                    ax.hist(data, bins=20, density=True, alpha=0.7, edgecolor='black', color='skyblue')
                    
                    # Add normal curve for comparison
                    if len(data) > 1:
                        mu, sigma = data.mean(), data.std()
                        x = np.linspace(data.min(), data.max(), 100)
                        normal_curve = stats.norm.pdf(x, mu, sigma)
                        ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal curve')
                    
                    # Add reference lines at -1, 0, +1
                    ax.axvline(-1, color='blue', linestyle='--', alpha=0.5, label='Biden/Liberal')
                    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
                    ax.axvline(1, color='red', linestyle='--', alpha=0.5, label='Trump/Conservative')
                    
                    ax.set_title(f'{var_name.replace("_", " ").title()}')
                    ax.set_xlabel('Bias Score')
                    ax.set_ylabel('Density')
                    ax.set_xlim(-1.2, 1.2)
                    if i == 0:  # Only show legend on first plot
                        ax.legend()
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = PROCESSED_DATA_DIR / "assumption_check_distributions.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved assumption check distribution plots to {plot_path}")
            
            plt.close()
        
        return assumption_results, clean_df
    
    # Run assumption checks
    assumption_results, analysis_df = check_assumptions(
        unified_df, 
        {**key_variables, **bias_markers}, 
        save_plots=save_plots
    )
    
    if len(analysis_df) < 10:
        logger.error("Insufficient data for correlation analysis after removing missing values")
        return
        
    logger.success(f"Assumption checking complete. Analysis sample: {len(analysis_df)} polls")
    
    # Compute pairwise correlations
    logger.info("Computing pairwise Pearson correlations...")
    
    def compute_pairwise_correlations(df, markers, outcomes):
        """Compute all pairwise correlations between markers and outcomes"""
        
        correlation_results = []
        
        for marker_name, marker_col in markers.items():
            for outcome_name, outcome_col in outcomes.items():
                
                # Get data for this pair, removing missing values
                marker_data = df[marker_col].dropna()
                outcome_data = df[outcome_col].dropna()
                
                # Find common indices (complete pairs)
                common_idx = marker_data.index.intersection(outcome_data.index)
                
                if len(common_idx) < 10:  # Minimum sample size for meaningful correlation
                    logger.warning(f"Insufficient data for {marker_name} vs {outcome_name}: {len(common_idx)} pairs")
                    continue
                
                # Extract paired data
                x = df.loc[common_idx, marker_col]
                y = df.loc[common_idx, outcome_col]
                
                # Compute Pearson correlation
                try:
                    r, p_value = stats.pearsonr(x, y)
                    
                    # Store results
                    correlation_results.append({
                        'marker': marker_name,
                        'outcome': outcome_name,
                        'r': r,
                        'p_value': p_value,
                        'n': len(common_idx),
                        'r_squared': r**2,
                        'marker_col': marker_col,
                        'outcome_col': outcome_col
                    })
                    
                except Exception as e:
                    logger.warning(f"Correlation failed for {marker_name} vs {outcome_name}: {e}")
                    continue
        
        return pd.DataFrame(correlation_results)
    
    # Compute focused correlations: bias markers vs key outcomes
    correlation_df = compute_pairwise_correlations(analysis_df, bias_markers, key_variables)
    
    if correlation_df.empty:
        logger.error("No correlations could be computed!")
        return
        
    logger.success(f"Computed {len(correlation_df)} pairwise correlations")
    
    # Apply multiple testing correction
    logger.info("Applying Benjamini-Hochberg FDR correction...")
    
    # Apply FDR correction to all p-values
    if len(correlation_df) > 1:
        rejected, p_adjusted, alpha_sidak, alpha_bonf = multi.multipletests(
            correlation_df['p_value'], 
            alpha=0.05, 
            method='fdr_bh'  # Benjamini-Hochberg procedure
        )
        
        correlation_df['p_adjusted'] = p_adjusted
        correlation_df['significant_fdr'] = rejected
        
        n_significant = rejected.sum()
        logger.success(f"FDR correction complete. {n_significant} of {len(correlation_df)} correlations remain significant")
    else:
        correlation_df['p_adjusted'] = correlation_df['p_value']
        correlation_df['significant_fdr'] = correlation_df['p_value'] < 0.05
    
    # Compute bootstrap confidence intervals
    logger.info("Computing bootstrap 95% confidence intervals...")
    
    def bootstrap_correlation_ci(x, y, n_bootstrap=1000, confidence=0.95):
        """Compute bootstrap confidence interval for correlation coefficient"""
        
        if len(x) != len(y) or len(x) < 10:
            return np.nan, np.nan
            
        correlations = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = resample(range(len(x)), random_state=None)
            x_boot = x.iloc[indices] if hasattr(x, 'iloc') else x[indices]
            y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            
            # Compute correlation for this bootstrap sample
            try:
                r_boot, _ = stats.pearsonr(x_boot, y_boot)
                if not np.isnan(r_boot):
                    correlations.append(r_boot)
            except:
                continue
        
        if len(correlations) < 100:  # Need sufficient bootstrap samples
            return np.nan, np.nan
            
        # Compute confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(correlations, lower_percentile)
        ci_upper = np.percentile(correlations, upper_percentile)
        
        return ci_lower, ci_upper
    
    # Compute CIs for significant correlations (to save computation time)
    ci_results = []
    significant_correlations = correlation_df[correlation_df['significant_fdr']]
    
    logger.info(f"Computing CIs for {len(significant_correlations)} significant correlations...")
    
    for _, row in significant_correlations.iterrows():
        # Get the data for this correlation
        common_idx = analysis_df[row['marker_col']].dropna().index.intersection(
            analysis_df[row['outcome_col']].dropna().index
        )
        
        if len(common_idx) >= 10:
            x = analysis_df.loc[common_idx, row['marker_col']]
            y = analysis_df.loc[common_idx, row['outcome_col']]
            
            ci_lower, ci_upper = bootstrap_correlation_ci(x, y)
            
            ci_results.append({
                'marker': row['marker'],
                'outcome': row['outcome'],
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })
    
    # Merge CI results back to main dataframe
    ci_df = pd.DataFrame(ci_results)
    if not ci_df.empty:
        correlation_df = correlation_df.merge(
            ci_df, on=['marker', 'outcome'], how='left'
        )
    else:
        correlation_df['ci_lower'] = np.nan
        correlation_df['ci_upper'] = np.nan
    
    # Interpret effect sizes
    logger.info("Interpreting effect sizes using Cohen's benchmarks...")
    
    def interpret_correlation_effect_size(r):
        """Interpret correlation effect size using Cohen's benchmarks"""
        r_abs = abs(r)
        if r_abs < 0.1:
            return 'negligible'
        elif r_abs < 0.3:
            return 'small'
        elif r_abs < 0.5:
            return 'medium'
        else:
            return 'large'
    
    correlation_df['effect_size'] = correlation_df['r'].apply(interpret_correlation_effect_size)
    
    # Generate results summary
    logger.info("Generating comprehensive results summary...")
    
    # Sort by effect size and significance
    correlation_df = correlation_df.sort_values(['significant_fdr', 'r_squared'], ascending=[False, False])
    
    # Create summary statistics
    summary_stats = {
        'total_correlations': int(len(correlation_df)),
        'significant_raw': int((correlation_df['p_value'] < 0.05).sum()),
        'significant_fdr': int(correlation_df['significant_fdr'].sum()),
        'large_effects': int((correlation_df['effect_size'] == 'large').sum()),
        'medium_effects': int((correlation_df['effect_size'] == 'medium').sum()),
        'small_effects': int((correlation_df['effect_size'] == 'small').sum()),
        'sample_size': int(len(analysis_df))
    }
    
    # Log key findings
    logger.success("=== CORRELATION ANALYSIS RESULTS ===")
    logger.info(f"Total correlations tested: {summary_stats['total_correlations']}")
    logger.info(f"Significant before FDR correction: {summary_stats['significant_raw']}")
    logger.info(f"Significant after FDR correction: {summary_stats['significant_fdr']}")
    logger.info(f"Effect sizes - Large: {summary_stats['large_effects']}, Medium: {summary_stats['medium_effects']}, Small: {summary_stats['small_effects']}")
    logger.info(f"Analysis sample size: {summary_stats['sample_size']} polls")
    
    # Display top significant correlations
    significant_results = correlation_df[correlation_df['significant_fdr']].head(10)
    
    if len(significant_results) > 0:
        logger.success("\n=== TOP SIGNIFICANT CORRELATIONS (FDR-corrected) ===")
        for _, row in significant_results.iterrows():
            ci_str = ""
            if not pd.isna(row.get('ci_lower')):
                ci_str = f", 95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            
            logger.info(
                f"{row['marker']} ↔ {row['outcome']}: "
                f"r = {row['r']:.3f} (p = {row['p_adjusted']:.4f}, "
                f"n = {row['n']}, effect = {row['effect_size']}){ci_str}"
            )
    else:
        logger.warning("No significant correlations found after FDR correction")
    
    # Save detailed results to CSV
    output_path = PROCESSED_DATA_DIR / "pearson_correlation_results.csv"
    correlation_df.to_csv(output_path, index=False)
    logger.success(f"Detailed correlation results saved to {output_path}")
    
    # Save summary statistics
    summary_path = PROCESSED_DATA_DIR / "correlation_analysis_summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.success(f"Summary statistics saved to {summary_path}")
    
    # Create correlation heatmap
    if save_plots and not correlation_df.empty:
        logger.info("Creating final correlation heatmap from computed results...")
        
        def create_final_correlation_heatmap(corr_results_df, markers_dict, outcomes_dict):
            """Create correlation heatmap with significance indicators."""
            
            # Create correlation matrix from results
            marker_names = list(markers_dict.keys())
            outcome_names = list(outcomes_dict.keys()) 
            
            # Initialize matrix with NaN
            corr_matrix = pd.DataFrame(index=outcome_names, columns=marker_names, dtype=float)
            
            # Fill matrix with computed correlations
            for _, row in corr_results_df.iterrows():
                marker = row['marker']
                outcome = row['outcome']
                r_value = row['r']
                
                if marker in marker_names and outcome in outcome_names:
                    corr_matrix.loc[outcome, marker] = r_value
            
            # Convert to numeric, replacing any remaining NaN
            corr_matrix = corr_matrix.astype(float)
            
            # Create readable labels  
            marker_labels = []
            for marker in marker_names:
                if marker == 'candidate_order':
                    marker_labels.append('Candidate Order')
                elif marker == 'formality_bias':
                    marker_labels.append('Formality Bias')
                elif marker == 'political_leaning':
                    marker_labels.append('Political Leaning')
                else:
                    marker_labels.append(marker.replace('_', ' ').title())
            
            outcome_labels = []
            for outcome in outcome_names:
                if outcome == 'author_bias':
                    outcome_labels.append('Author Bias')
                elif outcome == 'audience_bias':
                    outcome_labels.append('Audience Bias') 
                elif outcome == 'poll_outcome':
                    outcome_labels.append('Poll Outcome')
                else:
                    outcome_labels.append(outcome.replace('_', ' ').title())
            
            # Create the heatmap
            plt.figure(figsize=(8, 6))
            
            # Mask for significance - only show significant correlations with bold text
            significance_mask = pd.DataFrame(index=outcome_names, columns=marker_names, dtype=bool)
            for _, row in corr_results_df.iterrows():
                if row['marker'] in marker_names and row['outcome'] in outcome_names:
                    significance_mask.loc[row['outcome'], row['marker']] = row['significant_fdr']
            
            # Create annotations with significance indicators
            annot_matrix = corr_matrix.copy()
            for i, outcome in enumerate(outcome_names):
                for j, marker in enumerate(marker_names):
                    r_val = corr_matrix.loc[outcome, marker]
                    if pd.notna(r_val):
                        is_sig = significance_mask.loc[outcome, marker]
                        if is_sig:
                            annot_matrix.loc[outcome, marker] = f"{r_val:.3f}*"
                        else:
                            annot_matrix.loc[outcome, marker] = f"{r_val:.3f}"
                    else:
                        annot_matrix.loc[outcome, marker] = ""
            
            sns.heatmap(corr_matrix, annot=annot_matrix, fmt='', cmap='RdBu_r', center=0,
                       cbar_kws={"shrink": .8}, 
                       xticklabels=marker_labels, yticklabels=outcome_labels)
            
            plt.title('Pearson Correlations: Markers → Biases', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Bias Markers', fontweight='bold')
            plt.ylabel('Bias Values', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save final heatmap
            final_plot_path = PROCESSED_DATA_DIR / "final_correlation_heatmap.png"
            plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
            logger.success(f"Final correlation heatmap saved to {final_plot_path}")
            
            plt.close()
            
            return corr_matrix
        
        # Create final heatmap using computed results
        final_matrix = create_final_correlation_heatmap(correlation_df, bias_markers, key_variables)
        
        # Log the correlation matrix for verification
        logger.info("Final correlation matrix:")
        logger.info(f"\n{final_matrix}")
    
    logger.success("Pearson correlation analysis complete!")
    
    return correlation_df

if __name__ == "__main__":
    app()
