"""
Feature extraction for Twitter poll bias analysis.

This module transforms raw Twitter poll data into structured features for statistical analysis.
The primary research focus is understanding how candidate positioning and presentation
in Twitter polls influences voting behavior and outcomes.

All election-specific variables (candidate names, NLI hypotheses, regex patterns,
bias-direction labels, …) are loaded from bias_analysis.election_configs so that the
same pipeline can be reused across different elections worldwide.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import typing
from collections import defaultdict

from loguru import logger
from tqdm import tqdm
import typer

from bias_analysis.config import get_election_paths
from bias_analysis.election_configs import get_election_config
from bias_analysis.dataset import get_base_dataset
import ast

app = typer.Typer()

# Default election code used when --election is not provided on the CLI.
DEFAULT_ELECTION = "us20"


# ---------------------------------------------------------------------------
# Candidate name matching (election-agnostic)
# ---------------------------------------------------------------------------

def smart_candidate_match(candidate_text: str, election_config: dict) -> str:
    """
    Normalize candidate names handling variations, emojis, and common prefixes/suffixes.

    Args:
        candidate_text: Raw candidate text from poll option.
        election_config: Election configuration dict from election_configs.

    Returns:
        Normalized candidate name (one of the config's candidates, or 'Other').
    """
    # Build mapping from every known variation → canonical name
    candidate_mapping: dict[str, str] = {}
    for main_name, variations in election_config["candidate_match_rules"].items():
        for variation in variations:
            candidate_mapping[variation.strip()] = main_name

    if not candidate_text or not isinstance(candidate_text, str):
        return "Other"

    # Fast path: check for exact matches first (covers ~80% of cases)
    normalized_text = candidate_text.strip()
    if normalized_text in candidate_mapping:
        return candidate_mapping[normalized_text]

    # Fuzzy matching path: handle emoji-decorated and prefix/suffix variations
    cleaned_text = re.sub(r'[^\w\s\-\.\(\)\/]', ' ', normalized_text)

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
        r'\s+(?:20\d{2}|for president|for pres|presidency|administration|admin)$',
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

    for candidate_name, patterns in election_config["candidate_match_patterns"].items():
        for pattern in patterns:
            if re.search(pattern, cleaned_lower):
                return candidate_name

    return "Other"


@app.command()
def candidate_order(
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    """
    Extracts candidate positioning and voting patterns from Twitter polls.

    Output CSV contains columns:
    - {candidate}_position: Position in poll options (1, 2, 3, 4)
    - {candidate}_votes: Raw vote count
    - {candidate}_percentage: Vote percentage
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    candidates = ecfg["candidates"]
    all_candidates = candidates + ["Other"]

    logger.info(f"Starting candidate order feature extraction for [{election}]...")

    # Load the unified foundational dataset
    base_df = get_base_dataset(election=election)

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
        for candidate in all_candidates:
            poll_record[f'{candidate}_position'] = None
            poll_record[f'{candidate}_votes'] = None
            poll_record[f'{candidate}_percentage'] = None

        # Extract data from each poll option
        for option in options:
            candidate_name = option.get('label', '')
            position = option.get('position', 0)
            votes = option.get('votes', 0)
            vote_percentage = (votes / total_votes) * 100 if total_votes > 0 else 0

            normalized_name = smart_candidate_match(candidate_name, ecfg)

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
    output_path = paths.processed_dir / "candidate_order_features.csv"
    result_df.to_csv(output_path, index=False)

    # Log summary statistics for data quality assessment
    logger.success(f"Candidate order features saved to {output_path}")
    logger.info(f"Generated {len(result_df)} poll records")
    logger.info(f"Columns created: {len(result_df.columns)}")

    # Report candidate coverage to assess data quality
    for candidate in all_candidates:
        count = result_df[f'{candidate}_position'].notna().sum()
        logger.info(f"Polls with {candidate}: {count}")

    return result_df

@app.command()
def political_leaning(
    max_samples: int = typer.Option(None, help="Maximum number of samples to process from each dataset"),
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    """
    Analyzes political leaning indicators in poll text using facebook/bart-large-mnli.

    Args:
        max_samples: Limit processing to this many samples per dataset for testing
        election: Election code identifying the election config to use
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    candidates = ecfg["candidates"]
    hypotheses_cfg = ecfg["nli_hypotheses"]
    column_mapping = ecfg["hypothesis_column_mapping"]

    # Build ordered hypothesis list for the classifier
    hypothesis_keys = list(hypotheses_cfg.keys())
    hypotheses = [hypotheses_cfg[k] for k in hypothesis_keys]

    logger.info(f"Starting political leaning analysis for [{election}] using facebook/bart-large-mnli...")

    # Initialize the BART-MNLI model and tokenizer for natural language inference
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        logger.info("Loading facebook/bart-large-mnli model...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

        # Enable CUDA acceleration and half-precision if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 16-bit precision utilizes Tensor Cores on GPUs like the RTX 2080 Ti for speedups
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli",
            torch_dtype=torch_dtype
        )

        # Set model to evaluation mode to disable dropout and batch normalization training behavior
        model.eval()

        model.to(device)
        logger.info(f"Using device: {device} with precision: {torch_dtype}")

        logger.success("facebook/bart-large-mnli model loaded successfully")

    except ImportError:
        logger.error("transformers library not installed. Please install with: pip install transformers torch")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

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
                inputs = tokenizer(
                    text,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )

                # Move inputs to same device as model
                inputs = {k: v.to(device) for k, v in inputs.items()}

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
    base_df = get_base_dataset(election=election)

    if base_df.empty:
        logger.warning("Base dataset is empty. Cannot extract political leaning.")
        return

    sample_df = base_df.head(max_samples) if max_samples is not None and len(base_df) > max_samples else base_df
    if max_samples is not None:
        logger.info(f"Processing political leaning for {len(sample_df)} polls (limited to {max_samples} samples)...")
    else:
        logger.info(f"Processing political leaning for all {len(sample_df)} polls...")

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
        for candidate in candidates:
            poll_record[f'{candidate}_votes'] = None
            poll_record[f'{candidate}_percentage'] = None

        # Store political leaning scores from BART-MNLI classification
        # Map hypothesis key → column name → score
        for h_key, h_text in zip(hypothesis_keys, hypotheses):
            col_name = column_mapping[h_key]
            poll_record[col_name] = political_scores.get(h_text, 0.0)

        # Extract vote data from poll options to correlate with political leaning
        candidate_votes = {}
        for option in options:
            option_text = option.get('label', '')
            votes = option.get('votes', 0)
            vote_percentage = (votes / total_votes) * 100 if total_votes > 0 else 0

            normalized_candidate = smart_candidate_match(option_text, ecfg)

            if normalized_candidate in candidates:
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
    output_path = paths.processed_dir / "political_leaning_features.csv"
    result_df.to_csv(output_path, index=False)

    # Report analysis completion and summary statistics
    logger.success("Political leaning analysis completed!")
    logger.success(f"Political leaning features saved to {output_path}")
    logger.info(f"Generated {len(result_df)} poll records with political leaning scores")
    logger.info(f"Columns created: {len(result_df.columns)}")

    # Display statistical summary of political leaning classification results
    logger.info("Political leaning score statistics:")

    # Report mean and standard deviation for each political dimension
    score_cols = [column_mapping[k] for k in hypothesis_keys]
    for score_col in score_cols:
        if score_col in result_df.columns:
            mean_score = result_df[score_col].mean()
            std_score = result_df[score_col].std()
            logger.info(f"  {score_col}: mean={mean_score:.3f}, std={std_score:.3f}")

    # Report data coverage for correlation analysis
    for candidate in candidates:
        votes_count = result_df[f'{candidate}_votes'].notna().sum()
        logger.info(f"Polls with {candidate} votes: {votes_count}")

    return result_df

@app.command()
def formal_vs_informal(
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
    skip_ned: bool = typer.Option(False, "--skip-ned", help="Skip DBpedia Spotlight NED (faster, more false positives)"),
    ned_confidence: float = typer.Option(0.5, help="DBpedia Spotlight confidence threshold (0-1)"),
    ned_url: str = typer.Option(
        "https://api.dbpedia-spotlight.org/en/annotate",
        help="DBpedia Spotlight endpoint URL",
    ),
):
    """
    Extracts candidate call-outs from poll options/tweet texts and classifies
    their formality via the van den Berg et al. (2019) naming-form taxonomy.

    Pipeline per text:
      1. NER (spaCy PERSON) + regex fallback  -> raw mention spans
      2. NED (DBpedia Spotlight)              -> filter false positives
      3. Deduplication                        -> keep longest span per candidate
      4. Formality classification (van den Berg 2019 + extensions):
           6 TFNLN  Title+First+Last  ("President Donald Trump")
           5 TLN    Title+Last        ("President Trump")
           4 FNLN   First+Last        ("Donald Trump")
           3 LN     Last only         ("Trump")
           2 FN     First only        ("Donald")
           1 PET_NAME Pet/nickname    ("Donnie", "Sleepy Joe")
           0 ADJ_NAME Derog adj+name  ("Crooked Hillary")

    Output CSV per candidate:
      {C}_text, {C}_formality_score (0-6), {C}_formality_category,
      {C}_votes, {C}_percentage
    """
    import time

    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    candidates = ecfg["candidates"]
    candidate_naming_cfg = ecfg.get("candidate_naming", {})
    appellative_patterns_cfg = ecfg["appellative_patterns"]

    logger.info(f"Starting formal vs informal appellatives extraction for [{election}]...")
    if skip_ned:
        logger.warning("NED disabled (--skip-ned): false positives will not be filtered.")
    else:
        logger.info(f"NED endpoint: {ned_url}  confidence={ned_confidence}")

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

    # ── requests (optional, for NED) ─────────────────────────────────────────
    _requests = None
    if not skip_ned:
        try:
            import requests as _requests
        except ImportError:
            logger.warning("requests not installed — falling back to --skip-ned mode.")
            skip_ned = True

    # ── NED helper ────────────────────────────────────────────────────────────
    def disambiguate_with_dbpedia(text: str) -> dict:
        """
        Call DBpedia Spotlight on *text*; return {CanonicalName: surface_form}
        for any confirmed candidate. Filters false positives by matching the
        returned DBpedia URI against the expected URI in candidate_naming_cfg.
        Falls back to {} on any error.
        """
        if not text or not text.strip():
            return {}
        uri_to_cname = {
            info["dbpedia_uri"]: cname
            for cname, info in candidate_naming_cfg.items()
            if "dbpedia_uri" in info
        }
        confirmed = {}
        rejected = []
        for attempt in range(3):
            try:
                resp = _requests.post(
                    ned_url,
                    headers={"Accept": "application/json"},
                    data={"text": text, "confidence": ned_confidence,
                          "support": 10, "types": "DBpedia:Person"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    for res in resp.json().get("Resources", []):
                        uri     = res.get("@URI", "")
                        surface = res.get("@surfaceForm", "")
                        if uri in uri_to_cname:
                            cn = uri_to_cname[uri]
                            if cn not in confirmed or len(surface) > len(confirmed[cn]):
                                confirmed[cn] = surface
                        else:
                            rejected.append(surface)
                    return {"confirmed": confirmed, "rejected": rejected}
                elif resp.status_code == 503:
                    time.sleep(2 ** attempt)
                else:
                    return {"confirmed": {}, "rejected": []}
            except Exception:
                time.sleep(2 ** attempt)
        return {"confirmed": {}, "rejected": []}

    # ── Raw mention extraction (Step 1) ───────────────────────────────────────
    # Build set of valid preceding words globally to avoid doing it per-text
    _VALID_PRECEDING_WORDS = set()
    for cname in candidates:
        info = candidate_naming_cfg.get(cname, {})
        for t in info.get("titles", []):
            _VALID_PRECEDING_WORDS.update([w.lower() for w in t.split()])
        for d in info.get("derogatory_adjectives", []):
            _VALID_PRECEDING_WORDS.update([w.lower() for w in d.split()])
        first = info.get("first_name", "")
        if first:
            _VALID_PRECEDING_WORDS.add(first.lower())
        for a in info.get("alt_first_names", []):
            _VALID_PRECEDING_WORDS.update([w.lower() for w in a.split()])
        for p in info.get("pet_names", []):
            _VALID_PRECEDING_WORDS.update([w.lower() for w in p.split()])

    # This will be populated after loading the dataset
    text_to_doc = {}

    def extract_raw_mentions(text: str) -> dict:
        """spaCy NER + regex fallback; returns {cname_lower: [spans]}."""
        if not text or not isinstance(text, str):
            return {}
        mentions = {c.lower(): [] for c in candidates}
        doc = text_to_doc.get(text)
        if doc is None:
            doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            
            # Smart context expansion: only include previous tokens if relevant
            start_idx = ent.start
            while start_idx > 0 and start_idx >= ent.start - 3:
                prev_token = doc[start_idx - 1].text.lower().strip('.')
                if prev_token in _VALID_PRECEDING_WORDS:
                    start_idx -= 1
                else:
                    break

            span = doc[start_idx:ent.end].text.strip()
            span = re.sub(r'^(and|or|but|the|a|an|with|for|against|vs\.?|versus)\s+', '', span, flags=re.IGNORECASE)
            span = re.sub(r'\s+(and|or|but|with|for|against|vs\.?|versus|will|would|should|could)$', '', span, flags=re.IGNORECASE).strip()
            el = ent.text.lower()
            for cname in candidates:
                info     = candidate_naming_cfg.get(cname, {})
                last_l   = info.get("last_name", cname).lower()
                first_l  = info.get("first_name", "").lower()
                alt_first_l = [a.lower() for a in info.get("alt_first_names", [])]
                pets_l   = [p.lower() for p in info.get("pet_names", [])]
                titles_l = [t.lower().rstrip('.') for t in info.get("titles", [])]
                derogs_l = [d.lower() for d in info.get("derogatory_adjectives", [])]

                matched_by_last  = last_l  and last_l  in el
                matched_by_first = (first_l and first_l in el) or any(a and a in el for a in alt_first_l)

                if not (matched_by_last or matched_by_first):
                    continue

                # ── False-positive guard ────────────────────────────────────
                # If matched via last name, check whether a *different* proper
                # noun precedes it (e.g. "Hunter" in "Hunter Biden").
                # Valid preceding tokens: the candidate's own first name,
                # pet/nick names, title words, derogatory adjectives (still
                # relevant for ADJ_NAME), or no preceding word at all.
                if matched_by_last and not matched_by_first:
                    valid_preceding = (
                        {first_l}
                        | set([a.split()[0] for a in alt_first_l]) # e.g. 'joseph' from 'joseph r.'
                        | set(pets_l)
                        | set(titles_l)
                        | set(derogs_l)
                        | {""}   # nothing before the last name is fine
                    )
                    # Find the word immediately before the last name in the entity text
                    pre_match = re.search(
                        r'\b(\w+)\s+' + re.escape(last_l) + r'\b', el, re.I
                    )
                    if pre_match:
                        preceding_word = pre_match.group(1).lower().rstrip('.')
                        if preceding_word not in valid_preceding:
                            # "Hunter Biden", "Beau Biden", etc. → skip
                            continue
                # ────────────────────────────────────────────────────────────

                mentions[cname.lower()].append(span)

        # Regex fallback for any candidate still empty
        for cname in candidates:
            key = cname.lower()
            if not mentions[key]:
                for pattern in appellative_patterns_cfg.get(key, []):
                    m = re.search(pattern, text, re.IGNORECASE)
                    if m:
                        mentions[key].append(m.group(0).strip())
                        break

        return {k: v for k, v in mentions.items() if v}

    # ── Formality classifier (Step 4) ─────────────────────────────────────────
    _FORMALITY_SCALE = {
        "TFNLN": 6, "TLN": 5, "FNLN": 4, "LN": 3,
        "FN": 2, "PET_NAME": 1, "ADJ_NAME": 0,
    }

    def classify_formality_vandenberg(span: str, cname: str) -> tuple:
        """Return (score 0-6, category_code) for the given mention span."""
        if not span:
            return (_FORMALITY_SCALE["LN"], "LN")
        info     = candidate_naming_cfg.get(cname, {})
        first    = info.get("first_name", "")
        alt_first_names = info.get("alt_first_names", [])
        last     = info.get("last_name", cname)
        titles   = info.get("titles", [])
        pets     = info.get("pet_names", [])
        derogs   = info.get("derogatory_adjectives", [])

        def _any(words):
            return r'\b(?:' + '|'.join(re.escape(w) for w in words if w) + r')\b'

        has_last  = bool(last)   and bool(re.search(r'\b' + re.escape(last)  + r'\b', span, re.I))
        has_first = bool(first)  and bool(re.search(r'\b' + re.escape(first) + r'\b', span, re.I))
        if not has_first and alt_first_names:
            has_first = bool(re.search(_any(alt_first_names), span, re.I))
        has_title = bool(titles) and bool(re.search(_any(titles), span, re.I))

        # ADJ_NAME: derogatory adjective present
        if derogs and re.search(_any(derogs), span, re.I):
            return (_FORMALITY_SCALE["ADJ_NAME"], "ADJ_NAME")

        # PET_NAME: recognised nickname/pet name
        if pets:
            for pname in pets:
                if re.search(r'\b' + re.escape(pname) + r'\b', span, re.I):
                    return (_FORMALITY_SCALE["PET_NAME"], "PET_NAME")

        if has_title and has_first and has_last:
            return (_FORMALITY_SCALE["TFNLN"], "TFNLN")
        if has_title and has_last:
            return (_FORMALITY_SCALE["TLN"], "TLN")
        if has_first and has_last:
            return (_FORMALITY_SCALE["FNLN"], "FNLN")
        if has_last:
            return (_FORMALITY_SCALE["LN"], "LN")
        if has_first:
            return (_FORMALITY_SCALE["FN"], "FN")
        return (_FORMALITY_SCALE["LN"], "LN")

    # ── Load dataset ──────────────────────────────────────────────────────────
    base_df = get_base_dataset(election=election)
    if base_df.empty:
        logger.warning("Base dataset is empty. Cannot extract appellatives.")
        return

    # ── Pre-computation (Speed Optimization) ──────────────────────────────────
    all_texts = set()
    for _, row in base_df.iterrows():
        tweet_text = row.get('tweet_text', '') or ''
        if tweet_text:
            all_texts.add(tweet_text)
        try:
            options = ast.literal_eval(row['poll_options'])
            for opt in options:
                opt_text = opt.get('label', '')
                if opt_text:
                    all_texts.add(opt_text)
        except Exception:
            pass
    all_texts = list(all_texts)

    # Pre-compute spaCy docs using nlp.pipe
    logger.info(f"Batch processing {len(all_texts)} unique texts with spaCy...")
    for text, doc in zip(all_texts, nlp.pipe(all_texts, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"], batch_size=256)):
        text_to_doc[text] = doc

    # Pre-compute NED using ThreadPoolExecutor
    ned_cache: dict = {}   # text -> NED result
    if not skip_ned:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info(f"Batch querying DBpedia for {len(all_texts)} unique texts using 50 workers...")
        def safe_query(t):
            return t, disambiguate_with_dbpedia(t)
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_text = {executor.submit(safe_query, t): t for t in all_texts}
            for future in tqdm(as_completed(future_to_text), total=len(all_texts), desc="NED Queries"):
                t, res = future.result()
                ned_cache[t] = res

    appellative_data = []

    logger.info(f"Processing appellatives for {len(base_df)} polls...")

    for idx, row in tqdm(base_df.iterrows(), total=len(base_df), desc="Extracting appellatives"):
        try:
            options = ast.literal_eval(row['poll_options'])
        except Exception:
            continue
        if not options:
            continue

        total_votes = row['total_votes'] or 1

        poll_record = {
            'poll_id': str(row['tweet_id']),
            'total_votes': total_votes,
            'num_options': row['n_options'],
        }
        for candidate in candidates:
            poll_record[f'{candidate}_text']               = None
            poll_record[f'{candidate}_formality_score']    = None
            poll_record[f'{candidate}_formality_category'] = None
            poll_record[f'{candidate}_votes']              = None
            poll_record[f'{candidate}_percentage']         = None

        # NED on tweet text (cached)
        tweet_text = row.get('tweet_text', '') or ''
        if not skip_ned:
            if tweet_text not in ned_cache:
                ned_cache[tweet_text] = disambiguate_with_dbpedia(tweet_text)
            tweet_ned = ned_cache[tweet_text]
        else:
            tweet_ned = {"confirmed": {}, "rejected": []}

        tweet_raw = extract_raw_mentions(tweet_text)
        # Merge NED surface forms into tweet mentions
        tweet_mentions: dict = {}
        for cname in candidates:
            key = cname.lower()
            spans = list(tweet_raw.get(key, []))
            if cname in tweet_ned.get("confirmed", {}):
                spans.append(tweet_ned["confirmed"][cname])
            
            # Veto spans that overlap with DBpedia entities resolving to someone else
            if not skip_ned and tweet_ned.get("rejected"):
                rejected_list = tweet_ned["rejected"]
                filtered_spans = []
                for s in spans:
                    s_lower = s.lower()
                    is_rejected = any(r.lower() in s_lower or s_lower in r.lower() for r in rejected_list)
                    if not is_rejected:
                        filtered_spans.append(s)
                spans = filtered_spans
                
            tweet_mentions[key] = spans

        candidate_votes: dict = {}

        for option in options:
            option_text = option.get('label', '')
            votes       = option.get('votes', 0)
            vote_pct    = (votes / total_votes) * 100 if total_votes > 0 else 0

            norm = smart_candidate_match(option_text, ecfg)
            if norm not in candidates:
                continue

            if norm not in candidate_votes:
                candidate_votes[norm] = {'votes': votes, 'percentage': vote_pct}
            else:
                candidate_votes[norm]['votes']      += votes
                candidate_votes[norm]['percentage'] += vote_pct

            if poll_record[f'{norm}_text'] is not None:
                continue

            # Step 1: mentions from option label
            opt_raw = extract_raw_mentions(option_text)

            # NED on option label (if different from tweet text)
            if not skip_ned and option_text and option_text != tweet_text:
                if option_text not in ned_cache:
                    ned_cache[option_text] = disambiguate_with_dbpedia(option_text)
                opt_ned = ned_cache[option_text]
            elif option_text == tweet_text:
                opt_ned = tweet_ned
            else:
                opt_ned = {"confirmed": {}, "rejected": []}

            key = norm.lower()
            opt_spans = list(opt_raw.get(key, []))
            
            if norm in opt_ned.get("confirmed", {}):
                opt_spans.append(opt_ned["confirmed"][norm])
                
            if not skip_ned and opt_ned.get("rejected"):
                rejected_list = opt_ned["rejected"]
                filtered_spans = []
                for s in opt_spans:
                    s_lower = s.lower()
                    is_rejected = any(r.lower() in s_lower or s_lower in r.lower() for r in rejected_list)
                    if not is_rejected:
                        filtered_spans.append(s)
                opt_spans = filtered_spans

            all_spans: list = opt_spans + tweet_mentions.get(key, [])

            # NED filter: when a confirmed surface form exists, filter loosely
            if not skip_ned:
                confirmed_surface = tweet_ned.get("confirmed", {}).get(norm, "") or opt_ned.get("confirmed", {}).get(norm, "")
                if confirmed_surface:
                    filtered = [
                        s for s in all_spans
                        if confirmed_surface.lower() in s.lower()
                        or s.lower() in confirmed_surface.lower()
                    ]
                    if filtered:
                        all_spans = filtered

            if not all_spans:
                continue

            # Step 3: keep longest span (most informative)
            best_span = max(all_spans, key=len)

            # Step 4: classify
            score, category = classify_formality_vandenberg(best_span, norm)
            poll_record[f'{norm}_text']               = best_span
            poll_record[f'{norm}_formality_score']    = score
            poll_record[f'{norm}_formality_category'] = category

        for cname, vd in candidate_votes.items():
            poll_record[f'{cname}_votes']      = vd['votes']
            poll_record[f'{cname}_percentage'] = vd['percentage']

        appellative_data.append(poll_record)

    # ── Export ────────────────────────────────────────────────────────────────
    result_df = pd.DataFrame(appellative_data)
    if result_df.empty:
        logger.warning("No appellative data found in any dataset!")
        return

    output_path = paths.processed_dir / "formal_informal_appellatives.csv"
    result_df.to_csv(output_path, index=False)
    logger.success(f"Formal vs informal appellatives saved to {output_path}")
    logger.info(f"Generated {len(result_df)} poll records")

    category_order = ["TFNLN", "TLN", "FNLN", "LN", "FN", "PET_NAME", "ADJ_NAME"]
    for candidate in candidates:
        text_count = result_df[f'{candidate}_text'].notna().sum()
        logger.info(f"Polls with {candidate} mentions: {text_count}  |  votes: {result_df[f'{candidate}_votes'].notna().sum()}")
        if text_count > 0:
            cat_col = f'{candidate}_formality_category'
            if cat_col in result_df.columns:
                dist    = result_df[cat_col].value_counts()
                ordered = {cat: int(dist.get(cat, 0)) for cat in category_order}
                logger.info(f"  {candidate} formality distribution: {ordered}")
            score_col = f'{candidate}_formality_score'
            if score_col in result_df.columns:
                logger.info(f"  {candidate} mean formality score: {result_df[score_col].mean():.2f}/6")

    return result_df

@app.command()
def pearson_correlation(
    save_plots: bool = typer.Option(True, help="Save diagnostic plots to files"),
    max_rows: int = typer.Option(None, help="Limit dataset size for testing"),
    election: str = typer.Option(DEFAULT_ELECTION, help="Election code (e.g. 'us20')"),
):
    """
    Computes Pearson correlations between bias markers and bias outcomes with statistical controls.

    Args:
        save_plots: Save diagnostic plots to files
        max_rows: Limit dataset size for testing
        election: Election code identifying the election config to use
    """
    ecfg = get_election_config(election)
    paths = get_election_paths(election)
    candidates = ecfg["candidates"]
    column_mapping = ecfg["hypothesis_column_mapping"]
    bias_dir = ecfg["bias_direction"]
    labels = ecfg["label_aliases"]
    positive_candidate = bias_dir["positive"]
    negative_candidate = bias_dir["negative"]

    logger.info(f"Starting comprehensive Pearson correlation analysis for [{election}]...")

    # Import required statistical libraries
    try:
        import scipy.stats as stats
        import statsmodels.api as sm
        import statsmodels.stats.multitest as multi
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

    try:
        base_df = get_base_dataset(election=election, max_rows_per_file=max_rows)

        # Load marker CSVs dynamically
        markers_dir = paths.processed_dir
        candidate_order_df = pd.read_csv(markers_dir / "candidate_order_features.csv") if (markers_dir / "candidate_order_features.csv").exists() else pd.DataFrame()
        appellatives_df = pd.read_csv(markers_dir / "formal_informal_appellatives.csv") if (markers_dir / "formal_informal_appellatives.csv").exists() else pd.DataFrame()
        leaning_df = pd.read_csv(markers_dir / "political_leaning_features.csv") if (markers_dir / "political_leaning_features.csv").exists() else pd.DataFrame()

        # Merge markers into base_df
        unified_df = base_df.copy()

        # --- Candidate order merge ---
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
                missing_pos_mask = unified_df[neg_col].isna() | unified_df[pos_col].isna()
                unified_df.loc[missing_pos_mask, 'candidate_order'] = np.nan
            else:
                unified_df['candidate_order'] = np.nan
        else:
            unified_df['candidate_order'] = np.nan
            unified_df[pos_pct_col] = np.nan
            unified_df[neg_pct_col] = np.nan

        # --- Appellatives merge ---
        if not appellatives_df.empty:
            appellatives_df['poll_id'] = appellatives_df['poll_id'].astype(str)
            app_merge_cols = ['poll_id']
            for c in candidates:
                score_col = f'{c}_formality_score'
                # Support both new (formality_score) and legacy (_label) column names
                if score_col in appellatives_df.columns:
                    app_merge_cols.append(score_col)
                elif f'{c}_label' in appellatives_df.columns:
                    app_merge_cols.append(f'{c}_label')
            unified_df = unified_df.merge(appellatives_df[app_merge_cols],
                                          left_on='tweet_id', right_on='poll_id', how='left')
            for c in candidates:
                score_col = f'{c}_formality_score'
                if score_col in unified_df.columns:
                    # Normalize 0-6 scale to -1..+1: (score - 3) / 3
                    unified_df[f'{c.lower()}_formal_appellative'] = (
                        (unified_df[score_col].fillna(3.0) - 3.0) / 3.0
                    )
                elif f'{c}_label' in unified_df.columns:
                    # Legacy fallback
                    label_map = {'formal': 1.0, 'informal': -1.0, 'neutral': 0.0}
                    unified_df[f'{c.lower()}_formal_appellative'] = unified_df[f'{c}_label'].map(label_map)
                else:
                    unified_df[f'{c.lower()}_formal_appellative'] = np.nan
        else:
            for c in candidates:
                unified_df[f'{c.lower()}_formal_appellative'] = np.nan

        # --- Political leaning merge ---
        leaning_score_cols = list(column_mapping.values())

        if not leaning_df.empty:
            leaning_df['poll_id'] = leaning_df['poll_id'].astype(str)
            leaning_merge_cols = ['poll_id'] + [c for c in leaning_score_cols if c in leaning_df.columns]
            unified_df = unified_df.merge(leaning_df[leaning_merge_cols],
                                          left_on='tweet_id', right_on='poll_id', how='left')
        else:
            for col in leaning_score_cols:
                unified_df[col] = np.nan

        # Compute candidate shares
        if pos_pct_col in unified_df.columns:
            unified_df[f'{positive_candidate.lower()}_share'] = unified_df[pos_pct_col] / 100.0
        else:
            unified_df[f'{positive_candidate.lower()}_share'] = np.nan

        if neg_pct_col in unified_df.columns:
            unified_df[f'{negative_candidate.lower()}_share'] = unified_df[neg_pct_col] / 100.0
        else:
            unified_df[f'{negative_candidate.lower()}_share'] = np.nan

    except Exception as e:
        logger.error(f"Failed to load and merge unified dataset: {e}")
        return

    if unified_df.empty:
        logger.error("Unified dataset is empty!")
        return

    logger.success(f"Merged {len(unified_df)} polls with {len(unified_df.columns)} features")

    # Create author bias composite score
    pos_label = labels["positive_label"]
    neg_label = labels["negative_label"]
    logger.info(f"Computing author bias composite score (-1={neg_label}, +1={pos_label})...")

    positive_ideology_col = column_mapping.get("positive_ideology", "positive_ideology_score")
    candidate_A_support_col = column_mapping.get("candidate_A_support", "candidate_A_support_score")
    candidate_B_oppose_col = column_mapping.get("candidate_B_oppose", "candidate_B_oppose_score")
    negative_ideology_col = column_mapping.get("negative_ideology", "negative_ideology_score")
    candidate_B_support_col = column_mapping.get("candidate_B_support", "candidate_B_support_score")
    candidate_A_oppose_col = column_mapping.get("candidate_A_oppose", "candidate_A_oppose_score")

    pos_formal_col = f'{positive_candidate.lower()}_formal_appellative'
    neg_formal_col = f'{negative_candidate.lower()}_formal_appellative'

    def compute_author_bias(row):
        """
        Compute composite author bias from multiple indicators.

        Returns:
            float: -1 (negative candidate) to +1 (positive candidate)
        """
        bias_components = []

        # Author partisanship (normalized using tanh)
        if pd.notna(row.get('author_partisanship')):
            partisanship_normalized = np.tanh(row['author_partisanship'])
            bias_components.append(partisanship_normalized)

        # Candidate order bias (already scaled -1 to +1)
        if pd.notna(row.get('candidate_order')):
            bias_components.append(row['candidate_order'])

        # Formal appellatives
        if pd.notna(row.get(pos_formal_col)):
            bias_components.append(row[pos_formal_col])

        if pd.notna(row.get(neg_formal_col)):
            bias_components.append(-row[neg_formal_col])

        # Positive-direction sentiment scores (pro positive candidate)
        positive_scores = []
        for col in [positive_ideology_col, candidate_A_support_col, candidate_B_oppose_col]:
            if pd.notna(row.get(col)):
                positive_scores.append(row[col])

        if positive_scores:
            # Average positive scores and add to bias
            bias_components.append(np.mean(positive_scores))

        # Negative-direction sentiment scores (pro negative candidate)
        negative_scores = []
        for col in [negative_ideology_col, candidate_B_support_col, candidate_A_oppose_col]:
            if pd.notna(row.get(col)):
                negative_scores.append(row[col])

        if negative_scores:
            bias_components.append(-np.mean(negative_scores))

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
        Compute formality bias as the normalized score difference.

        Uses the van den Berg (2019) 7-level scale (0-6):
        formality_bias = (pos_score - neg_score) / 6.0  →  range [-1, +1]

        Returns:
            float: -1.0 (neg candidate more formal) to +1.0 (pos candidate more formal)
        """
        pos_formal = row.get(pos_formal_col, np.nan)
        neg_formal = row.get(neg_formal_col, np.nan)

        # pos_formal_col is already normalized to [-1,+1] via (score-3)/3
        # Just take the difference and clip
        if pd.isna(pos_formal) and pd.isna(neg_formal):
            return np.nan
        pos_v = float(pos_formal) if pd.notna(pos_formal) else 0.0
        neg_v = float(neg_formal) if pd.notna(neg_formal) else 0.0
        return float(np.clip(pos_v - neg_v, -1.0, 1.0))

    unified_df['formality_bias'] = unified_df.apply(compute_formality_bias, axis=1)

    # Log formality bias results
    formality_stats = unified_df['formality_bias'].describe()
    n_formality = unified_df['formality_bias'].notna().sum()

    logger.success(f"Formality bias computed for {n_formality}/{len(unified_df)} polls")
    logger.info(f"  Range: [{formality_stats['min']:.1f}, {formality_stats['max']:.1f}]")

    # Show formality distribution
    pos_only_formal = (unified_df['formality_bias'] == 1.0).sum()
    neg_only_formal = (unified_df['formality_bias'] == -1.0).sum()
    both_or_neither = (unified_df['formality_bias'] == 0.0).sum()

    logger.info(f"Formality distribution: {neg_only_formal} {negative_candidate}-only formal, {both_or_neither} both/neither, {pos_only_formal} {positive_candidate}-only formal")

    # Create audience bias and poll outcome measures
    logger.info("Computing audience bias and poll outcome measures...")

    def normalize_audience_bias(partisanship_score):
        """Normalize audience partisanship using tanh to [-1, +1] scale."""
        if pd.notna(partisanship_score):
            return np.tanh(partisanship_score)
        return np.nan

    unified_df['audience_bias'] = unified_df['audience_mean_partisanship'].apply(normalize_audience_bias)

    pos_share_col = f'{positive_candidate.lower()}_share'
    neg_share_col = f'{negative_candidate.lower()}_share'

    def compute_poll_outcome_bias(row):
        """
        Calculate poll outcome bias (-1=negative candidate win, +1=positive candidate win).

        Returns:
            float: -1.0 to +1.0 based on vote margin
        """
        pos_share = row.get(pos_share_col)
        neg_share = row.get(neg_share_col)

        if pd.notna(pos_share) and pd.notna(neg_share):
            # Calculate positive candidate advantage over negative candidate
            poll_bias = pos_share - neg_share
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
    logger.info(f"\nBias distribution (-1={neg_label}, +1={pos_label}):")
    for bias_name, bias_col in [('Author', 'author_bias'), ('Audience', 'audience_bias'), ('Poll Outcome', 'poll_outcome')]:
        if bias_col in unified_df.columns:
            data = unified_df[bias_col].dropna()
            if len(data) > 0:
                left = (data < -0.1).sum()
                neutral = ((data >= -0.1) & (data <= 0.1)).sum()
                right = (data > 0.1).sum()
                logger.info(f"  {bias_name}: {left} left, {neutral} neutral, {right} right")

    # Define key variables for focused correlation analysis
    # All variables normalized to -1 (negative candidate) to +1 (positive candidate) scale
    key_variables = {
        'author_bias': 'author_bias',        # Composite author bias
        'audience_bias': 'audience_bias',    # Normalized audience partisanship
        'poll_outcome': 'poll_outcome',      # Positive vs negative candidate poll advantage
    }

    # Bias markers to analyze
    bias_markers = {
        'candidate_order': 'candidate_order',          # M1: Position-based order effect
        'formality_bias': 'formality_bias',            # M2: Formality bias
        'political_leaning': positive_ideology_col,    # M3: Use positive ideology score directly
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
        """
        Checks statistical assumptions for Pearson correlation (Linearity, Normality, Outliers).

        Args:
            df (pd.DataFrame): Dataframe containing all variables
            variables_dict (dict): Keys=Variable descriptions, Values=Column names
            save_plots (bool): Save diagnostic plots

        Returns:
            tuple: (Assumption results dict, Cleaned DataFrame)
        """
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
            fig.suptitle(f'Distribution of Bias Measures (-1={neg_label}, +1={pos_label})', fontsize=14, fontweight='bold')

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
                    ax.axvline(-1, color='blue', linestyle='--', alpha=0.5, label=neg_label)
                    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
                    ax.axvline(1, color='red', linestyle='--', alpha=0.5, label=pos_label)

                    ax.set_title(f'{var_name.replace("_", " ").title()}')
                    ax.set_xlabel('Bias Score')
                    ax.set_ylabel('Density')
                    ax.set_xlim(-1.2, 1.2)
                    if i == 0:  # Only show legend on first plot
                        ax.legend()

            plt.tight_layout()

            if save_plots:
                plot_path = paths.figures_dir / "assumption_check_distributions.png"
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
            try:
                x_boot, y_boot = resample(x, y)
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
        'election': election,
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
    output_path = paths.processed_dir / "pearson_correlation_results.csv"
    correlation_df.to_csv(output_path, index=False)
    logger.success(f"Detailed correlation results saved to {output_path}")

    # Save summary statistics
    summary_path = paths.reports_dir / "correlation_analysis_summary.json"
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
            marker_labels = [m.replace('_', ' ').title() for m in marker_names]
            outcome_labels = [o.replace('_', ' ').title() for o in outcome_names]

            # Create the heatmap
            plt.figure(figsize=(8, 6))

            # Mask for significance - only show significant correlations with bold text
            significance_mask = pd.DataFrame(index=outcome_names, columns=marker_names, dtype=bool)
            for _, row in corr_results_df.iterrows():
                if row['marker'] in marker_names and row['outcome'] in outcome_names:
                    significance_mask.loc[row['outcome'], row['marker']] = row['significant_fdr']

            # Create annotations with significance indicators
            annot_matrix = corr_matrix.astype(object).copy()
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
            final_plot_path = paths.figures_dir / "final_correlation_heatmap.png"
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
