"""
Election-specific configuration registry.

Each election is identified by a short code (e.g. "us20") and defines all
election-specific variables that the analysis pipeline needs: candidate names,
name-matching rules, NLI hypotheses, appellative patterns, bias-direction
conventions, and display labels.

To add a new election, create a new dict following the same schema and
register it in ELECTION_CONFIGS.
"""


# ---------------------------------------------------------------------------
# US 2020 Presidential Election
# ---------------------------------------------------------------------------
_US20_CONFIG = {
    # Short code used on the CLI and in directory names
    "code": "us20",
    "display_name": "2020 US Presidential Election",

    # ── Candidates ────────────────────────────────────────────────────────
    # Ordered list of main candidates (used for column naming, iteration, …)
    "candidates": ["Trump", "Biden"],

    # ── smart_candidate_match rules ───────────────────────────────────────
    # Keys = canonical candidate name, values = known text variations.
    # Matching is tried in order: exact → cleaned → regex fallback.
    "candidate_match_rules": {
        "Trump": [
            "Trump", "trump", "TRUMP", "Donald Trump", "donald trump",
            "President Trump", "Donald J. Trump", "Donald J Trump",
        ],
        "Biden": [
            "Biden", "biden", "BIDEN", "Joe Biden", "joe biden",
            "President Biden", "Joe Biden (Democrat)", "Sleepy Joe",
            "sleepy joe",
        ],
    },

    # Regex fallback patterns (tried after exact / cleaned matching fails).
    # Each list is tried in order; first match wins.
    "candidate_match_patterns": {
        "Trump": [
            r"\btrump\b",
            r"\bdonald\b.*\btrump\b",
            r"\btrump\b.*\bdonald\b",
            r"\bpresident\s+trump\b",
            r"\bdonald\s+j\.?\s+trump\b",
        ],
        "Biden": [
            r"\bbiden\b",
            r"\bjoe\b.*\bbiden\b",
            r"\bbiden\b.*\bjoe\b",
            r"\bpresident\s+biden\b",
            r"\bjoseph\s+biden\b",
        ],
    },

    # ── NLI hypotheses (political_leaning command) ────────────────────────
    # key  → hypothesis text sent to the NLI model
    "nli_hypotheses": {
        "positive_ideology": "The text supports conservative politics",
        "negative_ideology": "The text supports liberal politics",
        "candidate_A_support": "The text praises Donald Trump",
        "candidate_B_support": "The text praises Joe Biden",
        "candidate_A_oppose": "The text criticizes Donald Trump",
        "candidate_B_oppose": "The text criticizes Joe Biden",
    },

    # key → output CSV column name
    "hypothesis_column_mapping": {
        "positive_ideology": "positive_ideology_score",
        "negative_ideology": "negative_ideology_score",
        "candidate_A_support": "candidate_A_support_score",
        "candidate_B_support": "candidate_B_support_score",
        "candidate_A_oppose": "candidate_A_oppose_score",
        "candidate_B_oppose": "candidate_B_oppose_score",
    },

    # ── Appellative regex patterns (formal_vs_informal command) ───────────
    # Used in extract_appellatives_from_text when spaCy NER misses a mention.
    "appellative_patterns": {
        "trump": [
            r"\b(?:president\s+)?(?:donald\s+)?(?:j\.?\s+)?trump\b",
            r"\btrump\b",
            r"\b(?:mr\.?\s+)?trump\b",
            r"\b(?:sleepy|crooked|crazy|orange|donnie)\b.*?\btrump\b",
            r"\btrump.*?\b(?:donald|don|donnie)\b",
        ],
        "biden": [
            r"\b(?:president\s+)?(?:joe\s+)?(?:joseph\s+)?biden\b",
            r"\bbiden\b",
            r"\b(?:mr\.?\s+)?biden\b",
            r"\b(?:sleepy|creepy|basement|dementia)\s+(?:joe\s+)?biden\b",
            r"\bbiden.*?\b(?:joe|joey|joseph)\b",
        ],
    },

    # ── Bias direction conventions ────────────────────────────────────────
    # "positive" candidate = +1 on the bias scale; "negative" = −1.
    "bias_direction": {
        "positive": "Trump",   # +1 end of scale
        "negative": "Biden",   # −1 end of scale
    },

    # ── Display labels (logs, plots, heatmaps) ───────────────────────────
    "label_aliases": {
        "positive_label": "Trump/Conservative",
        "negative_label": "Biden/Liberal",
    },

    # ── Plot colours per candidate ────────────────────────────────────────
    "candidate_colors": {
        "Trump": "tab:red",
        "Biden": "tab:blue",
    },

    # ── Election date (used by MRP dashboard to cap time axis) ───────────
    "election_date": "2020-11-03",

    # ── Campaign milestones (vertical markers on the MRP dashboard) ──────
    "milestones": [
        ("2020-06-18", "Tulsa effect: Pro-Trump Twitter mobilization artificially spikes poll numbers"),
        ("2020-07-04", "Biden's 'Soul of the Nation' July 4th message well-received amidst Trump struggles"),
        ("2020-07-15", "Trump replaces campaign manager Brad Parscale with Bill Stepien"),
        ("2020-09-09", "Woodward tapes released (Trump admits downplaying COVID-19)"),
        ("2020-09-29", "First presidential debate (Cleveland, Ohio)"),
        ("2020-10-07", "VP candidates debate (Salt Lake City, Utah)"),
        ("2020-10-15", "Second debate canceled (Trump COVID-19 positive)"),
        ("2020-10-22", "Second & last debate (Nashville, Tennessee)"),
    ],

    # ── Raw data file paths (relative to data/raw/<subdir>/) ─────────────
    # Each list must have the same length as data_source_names.
    "raw_data_paths": {
        "polls": [
            "Decahose/polls.jsonl",
            "vote/poll-vote-2020.jsonl",
            "voting/poll-voting-2020.jsonl",
        ],
        "partisanship": [
            "Decahose/inference/partisanship_scores_final.jsonl",
            "vote/inference/partisanship_scores_all_users.jsonl",
            "voting/inference/partisan-voting-2020.jsonl",
        ],
        "demographics": [
            "Decahose/inference/m3inf_output_final.jsonl",
            "vote/inference/m3inf_output_all_users.jsonl",
            "voting/inference/m3inf-voting-2020.jsonl",
        ],
        "retweeters": [
            "Decahose/retweeters.jsonl",
            "vote/retweet-vote-2020.jsonl",
            "voting/retweet-voting-2020.jsonl",
        ],
        "favoriters": [
            "Decahose/favoriters.jsonl",
            "vote/favorite-vote-2020.jsonl",
            "voting/favoriters-voting-2020.jsonl",
        ],
    },

    # Labels for each data source (matched by index to the path lists above)
    "data_source_names": ["decahose", "vote", "voting"],

    # ── Output sub-directory inside data/raw, data/processed, reports/… ───
    "processed_subdir": "us20",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ELECTION_CONFIGS = {
    "us20": _US20_CONFIG,
    # Future elections will be added here, e.g.:
    # "uk19": _UK19_CONFIG,
    # "fr22": _FR22_CONFIG,
}


def get_election_config(election_code: str) -> dict:
    """
    Return the configuration dict for the given election code.

    Args:
        election_code: Short identifier (e.g. "us20").

    Returns:
        dict with all election-specific settings.

    Raises:
        ValueError: If the election code is not registered.
    """
    if election_code not in ELECTION_CONFIGS:
        available = ", ".join(sorted(ELECTION_CONFIGS.keys()))
        raise ValueError(
            f"Unknown election code '{election_code}'. "
            f"Available elections: {available}"
        )
    return ELECTION_CONFIGS[election_code]


def list_available_elections() -> list[str]:
    """Return sorted list of registered election codes."""
    return sorted(ELECTION_CONFIGS.keys())
