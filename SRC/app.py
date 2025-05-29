"""
Main entry point for the call transcript analysis application
"""

import yaml
import argparse

from pathlib import Path
from processing import load_transcripts, postprocess_agent_predictions, generate_call_evaluation_insights
from agent import ActorAgent, CriticAgent
from evaluation import get_classification_report


def main():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_all',
                        action='store_true',
                        help='Specify whether to run all application modules')
    parser.add_argument('--run_classification',
                        action='store_true',
                        help='Specify whether to run the sentiment and outcome classification tasks')
    parser.add_argument('--run_critique',
                        action='store_true',
                        help='Specify whether to run the agent critique on sentiment and outcome classification tasks')
    parser.add_argument('--run_evaluation',
                        action='store_true',
                        help='Specify whether to run the evaluation of the classification outputs')
    args = parser.parse_args()

    print('Starting Call Transcripts Application')

    # Open the app config file and load in the app settings
    with open('SRC/config.yaml') as file:
        app_configuration = yaml.safe_load(file)

    # Load in agent prompts
    with open(project_root / app_configuration['agent_prompts_path']) as file:
        agent_prompts = yaml.safe_load(file)

    if args.run_classification or args.run_all:
        print('Running Classification')
        load_transcripts(data_read_path=project_root / app_configuration['data_read_path'],
                         data_save_path=project_root / app_configuration['data_save_path'],
                         limit=None)
        actor = ActorAgent(app_configuration=app_configuration, agent_prompts=agent_prompts)
        actor.compute_sentiment_on_dataset()
        actor.compute_outcome_on_dataset()

    if args.run_critique or args.run_all:
        print('Running Critique')
        critic = CriticAgent(app_configuration=app_configuration, agent_prompts=agent_prompts)
        critic.critique_sentiment_on_dataset()
        critic.critique_outcome_on_dataset()

    if args.run_evaluation or args.run_all:
        print('Running Evaluation')
        get_classification_report(app_configuration=app_configuration)

    print('Postprocessing agent conversation')
    postprocess_agent_predictions(app_configuration=app_configuration)

    print('Generating Insights')
    generate_call_evaluation_insights(app_configuration=app_configuration)

    print('Finished Running Call Transcripts Application')


if __name__ == '__main__':
    main()
