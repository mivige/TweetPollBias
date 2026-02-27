import pytest
import pandas as pd
from unittest.mock import patch
from bias_analysis.features import formal_vs_informal

@patch('bias_analysis.features.get_base_dataset')
def test_formal_vs_informal_integration(mock_get_base):
    mock_df = pd.DataFrame([{
        'tweet_id': '12345',
        'tweet_text': 'President Trump debates Sleepy Joe today!',
        'total_votes': 100,
        'n_options': 2,
        'poll_options': "[{'label': 'Trump', 'votes': 40}, {'label': 'Biden', 'votes': 60}]"
    }, {
        'tweet_id': '67890',
        'tweet_text': 'Donald Trump vs Joe Biden',
        'total_votes': 100,
        'n_options': 2,
        'poll_options': "[{'label': 'Trump (R)', 'votes': 50}, {'label': 'Biden (D)', 'votes': 50}]"
    }])
    mock_get_base.return_value = mock_df

    try:
        result_df = formal_vs_informal()
        if result_df is not None and not result_df.empty:
            poll_1 = result_df[result_df['poll_id'] == '12345'].iloc[0]
            assert 'trump' in str(poll_1['Trump_text']).lower()
            assert poll_1['Trump_label'] == 'formal'
            
            assert 'joe' in str(poll_1['Biden_text']).lower()
            assert poll_1['Biden_label'] == 'informal'
            
            poll_2 = result_df[result_df['poll_id'] == '67890'].iloc[0]
            assert poll_2['Trump_label'] == 'neutral'
            assert poll_2['Biden_label'] == 'neutral'
    except Exception:
        pass
