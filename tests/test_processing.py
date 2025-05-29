import os
import yaml
import pytest
import pandas as pd
from SRC import processing


@pytest.fixture()
def get_config():
    with open('SRC/config.yaml') as file:
        app_configuration = yaml.safe_load(file)
    return app_configuration


def test_load_transcripts(get_config):
    transcripts_df = processing.load_transcripts(data_read_path=get_config['data_read_path'])
    assert len(transcripts_df) == 200


def test_postprocessing(get_config):
    processing.postprocess_agent_predictions(get_config)
    predictions_df = pd.read_csv(get_config['predictions_path'])
    assert len(predictions_df) == 200


def test_generate_call_evaluation_insights(get_config):
    processing.generate_call_evaluation_insights(get_config)
    assert os.path.isfile(get_config['exhibits_path'] + '/transcript_count.png')
