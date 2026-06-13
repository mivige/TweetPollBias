import pytest
from bias_analysis.election_configs import (
    ELECTION_CONFIGS,
    get_election_config,
    list_available_elections,
)

KNOWN_ELECTIONS = ["us16", "us20", "us24"]

REQUIRED_KEYS = [
    "code",
    "display_name",
    "candidates",
    "candidate_match_rules",
    "candidate_match_patterns",
    "nli_hypotheses",
    "hypothesis_column_mapping",
    "candidate_naming",
    "appellative_patterns",
    "bias_direction",
    "label_aliases",
    "candidate_colors",
    "election_date",
    "milestones",
    "mrp",
    "raw_data_paths",
    "data_source_names",
    "processed_subdir",
]


def test_get_election_config_returns_correct_code():
    for code in KNOWN_ELECTIONS:
        assert get_election_config(code)["code"] == code


def test_get_election_config_raises_for_unknown_code():
    with pytest.raises(ValueError, match="Unknown election code"):
        get_election_config("xx99")


def test_error_message_lists_available_elections():
    with pytest.raises(ValueError, match="us20"):
        get_election_config("nope")


def test_list_available_elections_is_sorted():
    elections = list_available_elections()
    assert elections == sorted(elections)


def test_list_available_elections_contains_all_known():
    elections = list_available_elections()
    for code in KNOWN_ELECTIONS:
        assert code in elections


def test_list_available_elections_matches_registry():
    assert set(list_available_elections()) == set(ELECTION_CONFIGS.keys())


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_required_keys_present(election):
    cfg = get_election_config(election)
    for key in REQUIRED_KEYS:
        assert key in cfg, f"Missing key '{key}' in {election} config"


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_bias_direction_references_known_candidates(election):
    cfg = get_election_config(election)
    candidates = cfg["candidates"]
    assert cfg["bias_direction"]["positive"] in candidates
    assert cfg["bias_direction"]["negative"] in candidates
    assert cfg["bias_direction"]["positive"] != cfg["bias_direction"]["negative"]


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_actual_results_are_valid_probabilities(election):
    results = get_election_config(election)["mrp"]["actual_results"]
    for cand, share in results.items():
        assert 0.0 <= share <= 1.0, f"{election}: {cand} share {share!r} is out of [0, 1]"


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_actual_results_sum_to_one(election):
    results = get_election_config(election)["mrp"]["actual_results"]
    assert abs(sum(results.values()) - 1.0) < 0.01, (
        f"{election}: actual_results sum {sum(results.values()):.4f} ≠ 1"
    )


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_candidate_naming_keys_match_candidates(election):
    cfg = get_election_config(election)
    assert set(cfg["candidate_naming"].keys()) == set(cfg["candidates"])


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_data_source_names_length_matches_path_lists(election):
    cfg = get_election_config(election)
    n = len(cfg["data_source_names"])
    for key in ("polls", "partisanship", "demographics", "retweeters"):
        paths = cfg["raw_data_paths"][key]
        assert len(paths) == n, (
            f"{election}: '{key}' has {len(paths)} paths but "
            f"{n} data_source_names"
        )


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_candidate_match_rules_cover_all_candidates(election):
    cfg = get_election_config(election)
    for candidate in cfg["candidates"]:
        assert candidate in cfg["candidate_match_rules"], (
            f"{election}: no match rules for candidate '{candidate}'"
        )
        assert len(cfg["candidate_match_rules"][candidate]) > 0


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_nli_hypotheses_have_column_mapping(election):
    cfg = get_election_config(election)
    for key in cfg["nli_hypotheses"]:
        assert key in cfg["hypothesis_column_mapping"], (
            f"{election}: NLI hypothesis '{key}' has no column mapping"
        )


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_mrp_partisan_strata_sum_to_one(election):
    strata = get_election_config(election)["mrp"]["partisan_strata"]
    total = sum(strata.values())
    assert abs(total - 1.0) < 0.01, f"{election}: partisan_strata sum {total:.4f} ≠ 1"


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_mrp_ideological_strata_sum_to_one(election):
    strata = get_election_config(election)["mrp"]["ideological_strata"]
    total = sum(strata.values())
    assert abs(total - 1.0) < 0.01, f"{election}: ideological_strata sum {total:.4f} ≠ 1"


@pytest.mark.parametrize("election", KNOWN_ELECTIONS)
def test_candidate_colors_cover_all_candidates(election):
    cfg = get_election_config(election)
    for candidate in cfg["candidates"]:
        assert candidate in cfg["candidate_colors"]
