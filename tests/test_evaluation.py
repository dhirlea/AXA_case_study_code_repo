import yaml
import pytest
from SRC.evaluation import get_classification_report

TEST_DATASET_PATH = 'data/transcripts_dataset_test_sample.csv'


@pytest.fixture()
def get_app_config():
    with open('SRC/config.yaml') as file:
        app_configuration = yaml.safe_load(file)
    app_configuration['data_save_path'] = TEST_DATASET_PATH
    return app_configuration


def test_get_classification_report(get_app_config):
    get_classification_report(app_configuration=get_app_config)
