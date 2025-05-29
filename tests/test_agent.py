import yaml
import pytest
from SRC.agent import ActorAgent, CriticAgent

TEST_DATASET_PATH = 'data/transcripts_dataset_test_sample.csv'


@pytest.fixture()
def get_app_config():
    with open('SRC/config.yaml') as file:
        app_configuration = yaml.safe_load(file)
    app_configuration['data_save_path'] = TEST_DATASET_PATH
    return app_configuration


@pytest.fixture()
def get_actor_prompts():
    with open('prompts/agent_prompts.yaml') as file:
        actor_prompts = yaml.safe_load(file)
    return actor_prompts


def test_actor_compute_sentiment(get_app_config, get_actor_prompts):
    actor = ActorAgent(app_configuration=get_app_config, agent_prompts=get_actor_prompts)
    sample_transcripts_df = actor.compute_sentiment_on_dataset(save_flag=True)
    assert len(sample_transcripts_df) == 5


def test_actor_compute_outcome(get_app_config, get_actor_prompts):
    actor = ActorAgent(app_configuration=get_app_config, agent_prompts=get_actor_prompts)
    sample_transcripts_df = actor.compute_outcome_on_dataset(save_flag=True)
    assert len(sample_transcripts_df) == 5


def test_critic_sentiment(get_app_config, get_actor_prompts):
    critic = CriticAgent(app_configuration=get_app_config, agent_prompts=get_actor_prompts)
    sample_transcripts_df = critic.critique_sentiment_on_dataset(save_flag=True)
    assert len(sample_transcripts_df) == 5


def test_critic_outcome(get_app_config, get_actor_prompts):
    critic = CriticAgent(app_configuration=get_app_config, agent_prompts=get_actor_prompts)
    sample_transcripts_df = critic.critique_outcome_on_dataset(save_flag=True)
    assert len(sample_transcripts_df) == 5
