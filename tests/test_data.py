import pytest
from bias_analysis.features import smart_candidate_match

def test_smart_candidate_match():
    assert smart_candidate_match("Donald Trump") == "Trump"
    assert smart_candidate_match("Sleepy Joe") == "Biden"
    assert smart_candidate_match("Vote for Biden 💙") == "Biden"
    assert smart_candidate_match("Other candidate") == "Other"
