"""
Script hosting data ingestion and standardization utilities
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def load_transcripts(data_read_path: str, data_save_path: str = None, limit: int = None):
    """
    Utility function which loads in txt call transcripts and converts them into a Pandas DataFrame
    :param limit: option used to limit the number of transcripts to load in
    :param data_read_path:
    :param data_save_path:
    :return: df
    """
    data = []

    # Sort files by extracting numeric value from filename
    def extract_number(f):
        match = re.search(r'(\d+)', f)
        return int(match.group(1)) if match else -1

    # Get and sort .txt files by numeric part
    file_list = sorted(
        (f for f in os.listdir(data_read_path) if f.endswith('.txt')),
        key=extract_number
    )

    if limit:
        file_list = file_list[:limit]

    for filename in file_list:
        file_path = os.path.join(data_read_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Extract only the Member dialogue
            member_lines = re.findall(r"^Member:\s*(.*)", content, flags=re.MULTILINE)
            member_lines[0] = "Member: " + member_lines[0]
            member_dialogue = "\n".join(member_lines).strip()

            data.append({
                'file_name': filename,
                'raw_transcript': content,
                'member_dialogue': member_dialogue
            })

    df = pd.DataFrame(data)

    if data_save_path:
        df.to_csv(data_save_path, index=False)

    return df


def postprocess_agent_predictions(app_configuration: dict):
    """
    Utility function which selects final sentiment prediction and final outcome prediction from actor-critic dialogue.
    :param app_configuration:
    :return:
    """
    transcripts_df = pd.read_csv(app_configuration['data_save_path'])
    # take predictions from critic due to refined logic on top of actor's prediction
    transcripts_df['sentiment_prediction'] = transcripts_df['sentiment_with_critique']
    transcripts_df['outcome_prediction'] = transcripts_df['outcome_with_critique']
    transcripts_df[app_configuration['prediction_cols']].to_csv(app_configuration['predictions_path'], index=None)


def generate_call_evaluation_insights(app_configuration: dict):
    """
    Utility function to save insights chart from the sentiment and outcome agent predictions
    :param app_configuration:
    :return:
    """
    predictions_df = pd.read_csv(app_configuration['predictions_path'])
    predictions_df['transcript_length'] = predictions_df['raw_transcript'].apply(lambda x: len(x))
    predictions_by_sentiment_outcome_df = ((predictions_df.groupby(['sentiment_prediction', 'outcome_prediction'])
                                            .agg({'raw_transcript': 'count', 'transcript_length': 'mean'}))
                                           .reset_index()
                                           .rename(columns={'raw_transcript':
                                                            'transcript_count_by_sentiment_and_outcome',
                                                            'transcript_length':
                                                            'mean_transcript_length_by_sentiment_and_outcome'}))
    predictions_by_sentiment_df = ((predictions_df.groupby(['sentiment_prediction'])
                                    .agg({'raw_transcript': 'count', 'transcript_length': 'mean'})).reset_index()
                                   .rename(columns={'raw_transcript': 'transcript_count_by_sentiment',
                                                    'transcript_length': 'mean_transcript_length_by_sentiment'}))
    agg_predictions_df = predictions_by_sentiment_outcome_df.merge(predictions_by_sentiment_df,
                                                                   left_on=['sentiment_prediction'],
                                                                   right_on=['sentiment_prediction'],
                                                                   how='left')
    agg_predictions_df['proportion_count_by_outcome'] = \
        (agg_predictions_df['transcript_count_by_sentiment_and_outcome']
         / agg_predictions_df['transcript_count_by_sentiment'])

    # Grouped bar chart: Transcript count by sentiment and outcome
    pivot_counts = agg_predictions_df.pivot(index='sentiment_prediction', columns='outcome_prediction',
                                            values='transcript_count_by_sentiment_and_outcome')
    ax = pivot_counts.plot(kind='bar', figsize=(10, 6))
    plt.title("Transcript Count by Sentiment and Outcome")
    plt.ylabel("Count")
    plt.xlabel("Sentiment")
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3)
    plt.tight_layout()
    plt.savefig(app_configuration['exhibits_path'] + '/transcript_count.png')

    # Line plot: Mean transcript length by sentiment and outcome
    pivot_length = agg_predictions_df.pivot(index='sentiment_prediction', columns='outcome_prediction',
                                            values='mean_transcript_length_by_sentiment_and_outcome')
    ax = pivot_length.plot(marker='o', figsize=(10, 6))
    plt.title("Mean Transcript Length by Sentiment and Outcome")
    plt.ylabel("Mean Length")
    plt.xlabel("Sentiment")
    plt.xticks(rotation=0)
    for line in ax.get_lines():
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            ax.text(x, y + 3, f'{y:.0f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(app_configuration['exhibits_path'] + '/transcript_length.png')

    # Bar chart: Proportion count by outcome
    pivot_df = agg_predictions_df.pivot(index="outcome_prediction",
                                        columns="sentiment_prediction",
                                        values="proportion_count_by_outcome"
                                        )
    pivot_df = pivot_df.sort_index()
    ax = pivot_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Proportion Count by Outcome and Sentiment Prediction")
    plt.ylabel("Proportion of Transcripts")
    plt.xlabel("Outcome Prediction")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment Prediction")
    for container in ax.containers:
        # Generate labels by getting each bar's height and formatting as a percent
        labels = [f'{bar.get_height() * 100:.1f}%' for bar in container]
        ax.bar_label(container, labels=labels, padding=3)

    plt.tight_layout()
    plt.savefig(app_configuration['exhibits_path'] + '/outcome_proportions.png')
