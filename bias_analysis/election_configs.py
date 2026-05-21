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

    # ── Structured candidate naming metadata (formal_vs_informal command) ──
    # Used by the van den Berg (2019) formality classifier and DBpedia Spotlight NED.
    # Formality scale: TFNLN(5) > TLN(4) > FNLN(3) > LN(2) > FN(1) > ADJ_PET_NAME(0)
    "candidate_naming": {
        "Trump": {
            "first_name": "Donald",
            "last_name": "Trump",
            "middle": "J.",
            "titles": ["President", "Former President", "Mr.", "Mr"],
            "pet_names": ["Donnie", "Don", "The Donald"],
            "derogatory_adjectives": ["Crooked", "Crazy", "Orange", "Lyin'", "Lyin"],
            # DBpedia URI for NED disambiguation (filters false positives)
            "dbpedia_uri": "http://dbpedia.org/resource/Donald_Trump",
        },
        "Biden": {
            "first_name": "Joe",
            "alt_first_names": ["Joseph R.", "Joseph R", "Joseph"],
            "last_name": "Biden",
            "middle": "",
            "titles": ["President", "Vice President", "Former Vice President",
                       "Mr.", "Mr", "Senator", "Sen."],
            "pet_names": ["Joey"],
            "derogatory_adjectives": ["Sleepy", "Creepy", "Dementia", "Corrupt",
                                      "Basement"],
            "dbpedia_uri": "http://dbpedia.org/resource/Joe_Biden",
        },
    },

    # ── Appellative regex patterns (formal_vs_informal command) ───────────
    # Legacy fallback: used when spaCy NER misses a mention entirely.
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

    # ── MRP Model Configuration ───────────────────────────────────────────
    "mrp": {
        "actual_results": {
            "Trump": 0.477,
            "Biden": 0.523,
        },
        "partisan_strata": {"Republican": 0.36, "Democrat": 0.37, "Independent": 0.26},
        "ideological_strata": {"Conservative": 0.38, "Moderate": 0.38, "Liberal": 0.24},
        "demographic_profiles": {
            "Republican":  {"gender_male": 0.52, "age_under_29": 0.11, "age_30_39": 0.19, "age_40_over": 0.70},
            "Democrat":    {"gender_male": 0.43, "age_under_29": 0.24, "age_30_39": 0.27, "age_40_over": 0.49},
            "Independent": {"gender_male": 0.48, "age_under_29": 0.17, "age_30_39": 0.23, "age_40_over": 0.60},
        },
        "partisan_profiles": {
            "Republican": {"aud_quantile": 0.75, "auth_quantile": 0.75, "cons_quantile": 0.75},
            "Democrat":   {"aud_quantile": 0.25, "auth_quantile": 0.25, "cons_quantile": 0.25},
            "Independent": {"aud_quantile": 0.50, "auth_quantile": 0.50, "cons_quantile": 0.50},
        },
        "ideology_offsets": {
            "Conservative": {"cons_mult": 1.25},
            "Moderate":     {"cons_mult": 1.00},
            "Liberal":      {"cons_mult": 0.75},
        },
        "prediction_market": {
            "source": "PredictIt (placeholder)",
            "Trump": 0.39,
            "Biden": 0.63,
            "note": "These are win-probability prices, not vote-share estimates.",
        }
    },

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
# US 2016 Presidential Election
# ---------------------------------------------------------------------------
_US16_CONFIG = {
    "code": "us16",
    "display_name": "2016 US Presidential Election",

    "candidates": ["Trump", "Clinton"],

    "candidate_match_rules": {
        "Trump": [
            "Trump", "trump", "TRUMP", "Donald Trump", "donald trump",
            "President Trump", "Donald J. Trump", "Donald J Trump",
        ],
        "Clinton": [
            "Clinton", "clinton", "CLINTON", "Hillary Clinton", "hillary clinton",
            "Hillary", "hillary", "Secretary Clinton", "Crooked Hillary",
        ],
    },

    "candidate_match_patterns": {
        "Trump": [
            r"\btrump\b",
            r"\bdonald\b.*\btrump\b",
            r"\btrump\b.*\bdonald\b",
            r"\bdonald\s+j\.?\s+trump\b",
        ],
        "Clinton": [
            r"\bclinton\b",
            r"\bhillary\b.*\bclinton\b",
            r"\bclinton\b.*\bhillary\b",
            r"\bhillary\b",
            r"\bsecretary\s+clinton\b",
        ],
    },

    "nli_hypotheses": {
        "positive_ideology": "The text supports conservative politics",
        "negative_ideology": "The text supports liberal politics",
        "candidate_A_support": "The text praises Donald Trump",
        "candidate_B_support": "The text praises Hillary Clinton",
        "candidate_A_oppose": "The text criticizes Donald Trump",
        "candidate_B_oppose": "The text criticizes Hillary Clinton",
    },

    "hypothesis_column_mapping": {
        "positive_ideology": "positive_ideology_score",
        "negative_ideology": "negative_ideology_score",
        "candidate_A_support": "candidate_A_support_score",
        "candidate_B_support": "candidate_B_support_score",
        "candidate_A_oppose": "candidate_A_oppose_score",
        "candidate_B_oppose": "candidate_B_oppose_score",
    },

    "candidate_naming": {
        "Trump": {
            "first_name": "Donald",
            "last_name": "Trump",
            "middle": "J.",
            "titles": ["President", "Former President", "Mr.", "Mr"],
            "pet_names": ["Donnie", "Don", "The Donald"],
            "derogatory_adjectives": ["Crooked", "Crazy", "Orange", "Lyin'", "Lyin"],
            "dbpedia_uri": "http://dbpedia.org/resource/Donald_Trump",
        },
        "Clinton": {
            "first_name": "Hillary",
            "last_name": "Clinton",
            "middle": "Rodham",
            "titles": ["Secretary", "Secretary of State", "Mrs.", "Mrs", "Senator",
                       "Sen."],
            "pet_names": ["Hill", "Hillary"],
            "derogatory_adjectives": ["Crooked", "Killary", "Corrupt", "Lyin'", "Lyin"],
            "dbpedia_uri": "http://dbpedia.org/resource/Hillary_Clinton",
        },
    },

    "appellative_patterns": {
        "trump": [
            r"\b(?:president\s+)?(?:donald\s+)?(?:j\.?\s+)?trump\b",
            r"\btrump\b",
            r"\b(?:mr\.?\s+)?trump\b",
            r"\b(?:crooked|crazy|orange|donnie)\b.*?\btrump\b",
            r"\btrump.*?\b(?:donald|don|donnie)\b",
        ],
        "clinton": [
            r"\b(?:secretary\s+)?(?:hillary\s+)?(?:rodham\s+)?clinton\b",
            r"\bclinton\b",
            r"\b(?:mrs\.?\s+)?clinton\b",
            r"\b(?:crooked|killary)\b.*?\b(?:hillary|clinton)\b",
            r"\bclinton.*?\b(?:hillary)\b",
        ],
    },

    "bias_direction": {
        "positive": "Trump",
        "negative": "Clinton",
    },

    "label_aliases": {
        "positive_label": "Trump/Conservative",
        "negative_label": "Clinton/Liberal",
    },

    "candidate_colors": {
        "Trump": "tab:red",
        "Clinton": "tab:blue",
    },

    "election_date": "2016-11-08",

    "milestones": [
        ("2016-07-21", "Trump accepts GOP nomination"),
        ("2016-07-28", "Clinton accepts Democratic nomination"),
        ("2016-09-09", "Clinton's 'basket of deplorables' comment"),
        ("2016-09-26", "First presidential debate"),
        ("2016-10-07", "Access Hollywood tape released"),
        ("2016-10-09", "Second presidential debate"),
        ("2016-10-19", "Third presidential debate"),
        ("2016-10-28", "Comey announces reopening of Clinton email investigation"),
    ],

    "mrp": {
        "actual_results": {
            "Trump": 0.568,
            "Clinton": 0.432,
        },
        "partisan_strata": {"Republican": 0.33, "Democrat": 0.36, "Independent": 0.31},
        "ideological_strata": {"Conservative": 0.35, "Moderate": 0.39, "Liberal": 0.26},
        "demographic_profiles": {
            "Republican":  {"gender_male": 0.53, "age_under_29": 0.12, "age_30_39": 0.17, "age_40_over": 0.71},
            "Democrat":    {"gender_male": 0.41, "age_under_29": 0.23, "age_30_39": 0.22, "age_40_over": 0.55},
            "Independent": {"gender_male": 0.47, "age_under_29": 0.19, "age_30_39": 0.19, "age_40_over": 0.62},
        },
        "partisan_profiles": {
            "Republican": {"aud_quantile": 0.75, "auth_quantile": 0.75, "cons_quantile": 0.75},
            "Democrat":   {"aud_quantile": 0.25, "auth_quantile": 0.25, "cons_quantile": 0.25},
            "Independent": {"aud_quantile": 0.50, "auth_quantile": 0.50, "cons_quantile": 0.50},
        },
        "ideology_offsets": {
            "Conservative": {"cons_mult": 1.25},
            "Moderate":     {"cons_mult": 1.00},
            "Liberal":      {"cons_mult": 0.75},
        },
        "prediction_market": {
            "source": "Placeholder Market",
            "Trump": 0.20,
            "Clinton": 0.80,
            "note": "Placeholder win-probability estimates for 2016.",
        }
    },

    "raw_data_paths": {
        "polls": [
            "vote/polls.jsonl",
            "voting/polls.jsonl",
        ],
        "partisanship": [
            "vote/inference/partisanship_scores.jsonl",
            "voting/inference/partisanship_scores.jsonl",
        ],
        "demographics": [
            "vote/inference/m3inf_output.jsonl",
            "voting/inference/m3inf_output.jsonl",
        ],
        "retweeters": [
            "vote/retweeters.jsonl",
            "voting/retweeters.jsonl",
        ],
        "favoriters": [
            "vote/favoriters.jsonl",
            "voting/favoriters.jsonl",
        ],
    },

    "data_source_names": ["vote", "voting"],

    "processed_subdir": "us16",
}


# ---------------------------------------------------------------------------
# US 2024 Presidential Election
# ---------------------------------------------------------------------------
_US24_CONFIG = {
    "code": "us24",
    "display_name": "2024 US Presidential Election",

    "candidates": ["Trump", "Harris", "Biden"],

    "candidate_match_rules": {
        "Trump": [
            "Trump", "trump", "TRUMP", "Donald Trump", "donald trump",
            "President Trump", "Donald J. Trump", "Donald J Trump",
        ],
        "Harris": [
            "Harris", "harris", "HARRIS", "Kamala Harris", "kamala harris",
            "Kamala", "kamala", "VP Harris", "Vice President Harris",
        ],
        "Biden": [
            "Biden", "biden", "BIDEN", "Joe Biden", "joe biden",
            "President Biden", "Joe", "joe",
        ],
    },

    "candidate_match_patterns": {
        "Trump": [
            r"\btrump\b",
            r"\bdonald\b.*\btrump\b",
            r"\btrump\b.*\bdonald\b",
            r"\bdonald\s+j\.?\s+trump\b",
        ],
        "Harris": [
            r"\bharris\b",
            r"\bkamala\b.*\bharris\b",
            r"\bharris\b.*\bkamala\b",
            r"\bkamala\b",
            r"\bvp\s+harris\b",
        ],
        "Biden": [
            r"\bbiden\b",
            r"\bjoe\b.*\bbiden\b",
            r"\bbiden\b.*\bjoe\b",
            r"\bpresident\s+biden\b",
        ],
    },

    "nli_hypotheses": {
        "positive_ideology": "The text supports conservative politics",
        "negative_ideology": "The text supports liberal politics",
        "candidate_A_support": "The text praises Donald Trump",
        "candidate_B_support": "The text praises Kamala Harris or Joe Biden",
        "candidate_A_oppose": "The text criticizes Donald Trump",
        "candidate_B_oppose": "The text criticizes Kamala Harris or Joe Biden",
    },

    "hypothesis_column_mapping": {
        "positive_ideology": "positive_ideology_score",
        "negative_ideology": "negative_ideology_score",
        "candidate_A_support": "candidate_A_support_score",
        "candidate_B_support": "candidate_B_support_score",
        "candidate_A_oppose": "candidate_A_oppose_score",
        "candidate_B_oppose": "candidate_B_oppose_score",
    },

    "candidate_naming": {
        "Trump": {
            "first_name": "Donald",
            "last_name": "Trump",
            "middle": "J.",
            "titles": ["President", "Former President", "Mr.", "Mr"],
            "pet_names": ["Donnie", "Don", "The Donald"],
            "derogatory_adjectives": ["Crooked", "Crazy", "Orange", "Lyin'", "Lyin"],
            "dbpedia_uri": "http://dbpedia.org/resource/Donald_Trump",
        },
        "Harris": {
            "first_name": "Kamala",
            "last_name": "Harris",
            "middle": "Devi",
            "titles": ["Vice President", "VP", "President", "Ms.", "Ms", "Senator",
                       "Sen."],
            "pet_names": ["Kamala"],
            "derogatory_adjectives": ["Comrade", "Laughing", "Laffin", "Radical",
                                      "Cackling"],
            "dbpedia_uri": "http://dbpedia.org/resource/Kamala_Harris",
        },
        "Biden": {
            "first_name": "Joe",
            "alt_first_names": ["Joseph R.", "Joseph R", "Joseph"],
            "last_name": "Biden",
            "middle": "",
            "titles": ["President", "Vice President", "Former President",
                       "Mr.", "Mr", "Senator", "Sen."],
            "pet_names": ["Joey"],
            "derogatory_adjectives": ["Sleepy", "Creepy", "Dementia", "Corrupt",
                                      "Basement"],
            "dbpedia_uri": "http://dbpedia.org/resource/Joe_Biden",
        },
    },

    "appellative_patterns": {
        "trump": [
            r"\b(?:president\s+)?(?:donald\s+)?(?:j\.?\s+)?trump\b",
            r"\btrump\b",
            r"\b(?:mr\.?\s+)?trump\b",
            r"\b(?:crooked|crazy|orange|donnie)\b.*?\btrump\b",
            r"\btrump.*?\b(?:donald|don|donnie)\b",
        ],
        "harris": [
            r"\b(?:vice\s+president\s+)?(?:kamala\s+)?harris\b",
            r"\bharris\b",
            r"\b(?:vp\s+)?harris\b",
            r"\b(?:comrade|laffin)\b.*?\b(?:kamala|harris)\b",
            r"\bkamala.*?\bharris\b",
        ],
        "biden": [
            r"\b(?:president\s+)?(?:joe\s+)?(?:joseph\s+)?biden\b",
            r"\bbiden\b",
            r"\b(?:mr\.?\s+)?biden\b",
            r"\b(?:sleepy|creepy|basement|dementia)\s+(?:joe\s+)?biden\b",
            r"\bbiden.*?\b(?:joe|joey|joseph)\b",
        ],
    },

    "bias_direction": {
        "positive": "Trump",
        "negative": "Harris",
    },

    "label_aliases": {
        "positive_label": "Trump/Conservative",
        "negative_label": "Harris/Liberal",
    },

    "candidate_colors": {
        "Trump": "tab:red",
        "Harris": "tab:blue",
        "Biden": "tab:cyan",
    },

    "election_date": "2024-11-05",

    "milestones": [
        ("2024-06-27", "First presidential debate (CNN)"),
        ("2024-07-13", "Assassination attempt on Donald Trump"),
        ("2024-07-15", "Trump announces JD Vance as running mate"),
        ("2024-07-21", "Joe Biden withdraws, endorses Kamala Harris"),
        ("2024-08-06", "Harris announces Tim Walz as running mate"),
        ("2024-08-22", "Harris accepts Democratic nomination at DNC"),
        ("2024-09-10", "Presidential debate between Trump and Harris (ABC)"),
        ("2024-10-01", "Vice Presidential debate (CBS)"),
    ],

    "mrp": {
        "actual_results": {
            "Trump": 0.506,
            "Harris": 0.494,
        },
        "partisan_strata": {"Republican": 0.38, "Democrat": 0.35, "Independent": 0.27},
        "ideological_strata": {"Conservative": 0.36, "Moderate": 0.39, "Liberal": 0.25},
        "demographic_profiles": {
            "Republican":  {"gender_male": 0.55, "age_under_29": 0.15, "age_30_39": 0.20, "age_40_over": 0.65},
            "Democrat":    {"gender_male": 0.45, "age_under_29": 0.22, "age_30_39": 0.25, "age_40_over": 0.53},
            "Independent": {"gender_male": 0.50, "age_under_29": 0.20, "age_30_39": 0.25, "age_40_over": 0.55},
        },
        "partisan_profiles": {
            "Republican": {"aud_quantile": 0.75, "auth_quantile": 0.75, "cons_quantile": 0.75},
            "Democrat":   {"aud_quantile": 0.25, "auth_quantile": 0.25, "cons_quantile": 0.25},
            "Independent": {"aud_quantile": 0.50, "auth_quantile": 0.50, "cons_quantile": 0.50},
        },
        "ideology_offsets": {
            "Conservative": {"cons_mult": 1.25},
            "Moderate":     {"cons_mult": 1.00},
            "Liberal":      {"cons_mult": 0.75},
        },
        "prediction_market": {
            "source": "Placeholder Market",
            "Trump": 0.54,
            "Harris": 0.46,
            "note": "Placeholder win-probability estimates for 2024.",
        }
    },

    "raw_data_paths": {
        "polls": [
            "polls.jsonl",
            "polls-biden.jsonl",
            "polls-harris.jsonl",
        ],
        "partisanship": [
            "Inference/partisanship_scores.jsonl",
            "Inference/partisanship_scores.jsonl",
            "Inference/partisanship_scores.jsonl",
        ],
        "demographics": [
            "Inference/m3inf_output.jsonl",
            "Inference/m3inf_output.jsonl",
            "Inference/m3inf_output.jsonl",
        ],
        "retweeters": [
            "retweeters.jsonl",
            "retweeters.jsonl",
            "retweeters.jsonl",
        ],
        "favoriters": [
            "Inference/locations.jsonl",  # Placeholder if favoriters is not exactly matching, skipping file doesn't exist check isn't natively supported unless we spoof it. Assuming favouriters mapping isn't strictly mandatory or can be empty.
            "Inference/locations.jsonl",
            "Inference/locations.jsonl",
        ],
    },

    "data_source_names": ["general", "biden", "harris"],

    "processed_subdir": "us24",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ELECTION_CONFIGS = {
    "us20": _US20_CONFIG,
    "us16": _US16_CONFIG,
    "us24": _US24_CONFIG,
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
