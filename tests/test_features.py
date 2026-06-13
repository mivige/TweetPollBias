import pytest
from bias_analysis.election_configs import get_election_config
from bias_analysis.features import smart_candidate_match

US20 = get_election_config("us20")
US16 = get_election_config("us16")
US24 = get_election_config("us24")


# ---------------------------------------------------------------------------
# Exact matches from candidate_match_rules
# ---------------------------------------------------------------------------


class TestExactMatches:
    def test_canonical_name_matches(self):
        assert smart_candidate_match("Trump", US20) == "Trump"
        assert smart_candidate_match("Biden", US20) == "Biden"

    def test_lowercase_matches(self):
        assert smart_candidate_match("trump", US20) == "Trump"
        assert smart_candidate_match("biden", US20) == "Biden"

    def test_uppercase_matches(self):
        assert smart_candidate_match("TRUMP", US20) == "Trump"
        assert smart_candidate_match("BIDEN", US20) == "Biden"

    def test_full_name_matches(self):
        assert smart_candidate_match("Donald Trump", US20) == "Trump"
        assert smart_candidate_match("Joe Biden", US20) == "Biden"

    def test_us16_candidates(self):
        assert smart_candidate_match("Clinton", US16) == "Clinton"
        assert smart_candidate_match("Hillary Clinton", US16) == "Clinton"
        assert smart_candidate_match("Hillary", US16) == "Clinton"

    def test_us24_candidates(self):
        assert smart_candidate_match("Harris", US24) == "Harris"
        assert smart_candidate_match("Kamala Harris", US24) == "Harris"
        assert smart_candidate_match("Kamala", US24) == "Harris"


# ---------------------------------------------------------------------------
# Fuzzy matching: prefix / suffix stripping
# ---------------------------------------------------------------------------


class TestPrefixAndSuffixStripping:
    def test_vote_for_prefix_stripped(self):
        assert smart_candidate_match("Vote for Trump", US20) == "Trump"
        assert smart_candidate_match("vote for Biden", US20) == "Biden"

    def test_year_suffix_stripped(self):
        assert smart_candidate_match("Trump 2020", US20) == "Trump"
        assert smart_candidate_match("Biden 2024", US20) == "Biden"

    def test_for_president_suffix_stripped(self):
        assert smart_candidate_match("Biden for president", US20) == "Biden"

    def test_party_suffix_stripped(self):
        assert smart_candidate_match("Biden (Democrat)", US20) == "Biden"
        assert smart_candidate_match("Trump (Republican)", US20) == "Trump"


# ---------------------------------------------------------------------------
# Emoji / special character handling
# ---------------------------------------------------------------------------


class TestEmojiAndSpecialChars:
    def test_flag_emoji_before_name(self):
        assert smart_candidate_match("🇺🇸 Trump", US20) == "Trump"

    def test_checkmark_emoji_before_name(self):
        assert smart_candidate_match("✅ Biden", US20) == "Biden"

    def test_multiple_emojis(self):
        assert smart_candidate_match("🔥🔥 Trump 🔥🔥", US20) == "Trump"


# ---------------------------------------------------------------------------
# Regex fallback matching
# ---------------------------------------------------------------------------


class TestRegexFallback:
    def test_partial_name_with_title(self):
        assert smart_candidate_match("President Trump", US20) == "Trump"

    def test_first_and_last_name_order_variation(self):
        assert smart_candidate_match("donald trump", US20) == "Trump"

    def test_harris_regex_via_vp_title(self):
        assert smart_candidate_match("VP Harris", US24) == "Harris"

    def test_secretary_clinton_regex(self):
        assert smart_candidate_match("Secretary Clinton", US16) == "Clinton"


# ---------------------------------------------------------------------------
# Unknown / "Other" cases
# ---------------------------------------------------------------------------


class TestOtherCategory:
    def test_completely_unknown_name_returns_other(self):
        assert smart_candidate_match("Obama", US20) == "Other"

    def test_empty_string_returns_other(self):
        assert smart_candidate_match("", US20) == "Other"

    def test_none_returns_other(self):
        assert smart_candidate_match(None, US20) == "Other"  # type: ignore[arg-type]

    def test_whitespace_only_returns_other(self):
        assert smart_candidate_match("   ", US20) == "Other"

    def test_random_text_returns_other(self):
        assert smart_candidate_match("Neither candidate", US20) == "Other"


# ---------------------------------------------------------------------------
# Election-specificity: same input, different configs
# ---------------------------------------------------------------------------


class TestElectionSpecificity:
    def test_clinton_not_matched_in_us20(self):
        assert smart_candidate_match("Clinton", US20) == "Other"

    def test_harris_not_matched_in_us20(self):
        assert smart_candidate_match("Harris", US20) == "Other"

    def test_biden_not_matched_in_us16(self):
        assert smart_candidate_match("Biden", US16) == "Other"

    def test_trump_matched_across_all_elections(self):
        for cfg in (US16, US20, US24):
            assert smart_candidate_match("Trump", cfg) == "Trump"
