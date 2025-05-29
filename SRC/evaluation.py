"""
Script hosting evaluation functions for the transcript sentiment classification and outcome classification tasks
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

project_root = Path(__file__).resolve().parents[1]


def get_classification_report(app_configuration: dict):
    """
    Util function reading in actor predictions and critic evaluation and returns a classification report
    :param app_configuration:
    :return:
    """
    transcripts_df = pd.read_csv(project_root / app_configuration['data_save_path'])
    sentiment_report = classification_report(transcripts_df['sentiment_with_critique'],
                                             transcripts_df['sentiment'])
    print('Agent agreement report on call sentiment classification task')
    print(sentiment_report)
    outcome_report = classification_report(transcripts_df['outcome_with_critique'],
                                           transcripts_df['outcome'])
    print('Agent agreement report on call outcome classification task')
    print(outcome_report)
