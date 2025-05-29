"""
Script hosting agentic classes and methods for the actor and critic agents. Actor computes call sentiment and
call outcome. Critic independently validates sentiment and outcome, used to evaluate Actor performance.
"""

import re
import torch
import pandas as pd
from pathlib import Path
from transformers import pipeline

project_root = Path(__file__).resolve().parents[1]


# Create a base agent class to reduce duplication
class Agent:
    """
    Baseline agent class instantiating all attributes to be reused by the agents
    """

    def __init__(self, app_configuration: dict, agent_prompts: dict):
        self.app_configuration = app_configuration
        self.agent_prompts = agent_prompts
        self.pipeline = pipeline(task=self.app_configuration['sentiment_pipeline']['task'],
                                 model=self.app_configuration['model_id'],
                                 torch_dtype=torch.bfloat16,
                                 device_map=self.app_configuration['sentiment_pipeline']['device_map'],
                                 temperature=self.app_configuration['sentiment_pipeline']['temperature'],
                                 return_full_text=self.app_configuration['sentiment_pipeline']['return_full_text'],
                                 max_new_tokens=self.app_configuration['sentiment_pipeline']['max_new_tokens'])
        self._load_data()

    def _load_data(self):
        """
        Private method loading in call transcripts
        """
        try:
            self.transcripts_df = pd.read_csv(
                project_root / self.app_configuration['data_save_path']
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Transcript data not found: {e}")


class ActorAgent(Agent):
    """
    Class used to perform sentiment analysis and outcome determination on customer support call transcripts
    """

    def __init__(self, app_configuration: dict, agent_prompts: dict):
        super().__init__(app_configuration, agent_prompts)

    def compute_sentiment_on_dataset(self, save_flag: bool = True) -> pd.DataFrame:
        """
        Method used to compute sentiment scores across the transcript dataset
        :return:
        """

        def get_sentiment(transcript_str):
            sentiment_answer = self.pipeline(f"{self.agent_prompts['member_sentiment_prompt']}"
                                             f"{transcript_str} \n\n"
                                             f"SENTIMENT: ")[0]['generated_text']
            match = re.search(r'SENTIMENT\s*:\s*(POSITIVE|NEUTRAL|NEGATIVE)', sentiment_answer, re.IGNORECASE)
            if match:
                return match.group(1)
            else:
                return 'UNKNOWN'

        # Use member dialogue only to determine customer sentiment during the call
        self.transcripts_df['sentiment'] = self.transcripts_df['member_dialogue'].map(lambda x: get_sentiment(x))

        if save_flag:
            self.transcripts_df.to_csv(project_root / self.app_configuration['data_save_path'], index=False)

        return self.transcripts_df

    def compute_outcome_on_dataset(self, save_flag: bool = True) -> pd.DataFrame:
        """
        Method used to compute the outcome across the transcript dataset
        :return:
        """

        def get_outcome(transcript_str):
            outcome_answer = self.pipeline(f"{self.agent_prompts['member_outcome_prompt']}"
                                           f"{transcript_str} \n\n"
                                           f"CONCLUSION: ")[0]['generated_text']

            match = re.search(r"\bCONCLUSION:\s*(ISSUE RESOLVED|FOLLOW-UP ACTION NEEDED)\b",
                              outcome_answer, re.IGNORECASE)
            if match:
                return match.group(1)
            else:
                return 'UNKNOWN'

        # Use entire call transcript to determine actual outcome of the call, as often times the support representative
        # suggests a follow-up action on the call and the customer simply acknowledges this
        self.transcripts_df['outcome'] = self.transcripts_df['raw_transcript'].map(lambda x: get_outcome(x))
        if save_flag:
            self.transcripts_df.to_csv(project_root / self.app_configuration['data_save_path'], index=False)

        return self.transcripts_df


class CriticAgent(Agent):
    """
    Class used to perform evaluation of Actor predicted call sentiment and call outcome
    """

    def __init__(self, app_configuration: dict, agent_prompts: dict):
        super().__init__(app_configuration, agent_prompts)

    def critique_sentiment_on_dataset(self, save_flag: bool = True) -> pd.DataFrame:
        """
        Method used to compute sentiment scores across the transcript dataset
        :return:
        """

        def critique_sentiment(transcript_str, sentiment_str):
            sentiment_answer = self.pipeline(f"{self.agent_prompts['sentiment_critique_prompt_part1']}"
                                             f"{sentiment_str}\n"
                                             f"{self.agent_prompts['sentiment_critique_prompt_part2']}"
                                             f"{transcript_str}\n\n "
                                             f"SENTIMENT: ")[0]['generated_text']
            match = re.search(r'SENTIMENT\s*:\s*(POSITIVE|NEUTRAL|NEGATIVE)', sentiment_answer, re.IGNORECASE)
            if match:
                return match.group(1)
            else:
                return 'UNKNOWN'

        # Use member dialogue only to determine customer sentiment during the call
        self.transcripts_df['sentiment_with_critique'] = (self.transcripts_df
                                                          .apply(lambda row: critique_sentiment(row['member_dialogue'],
                                                                                                row['sentiment']),
                                                                 axis=1))

        if save_flag:
            self.transcripts_df.to_csv(project_root / self.app_configuration['data_save_path'], index=False)

        return self.transcripts_df

    def critique_outcome_on_dataset(self, save_flag: bool = True) -> pd.DataFrame:
        """
        Method used to compute the outcome across the transcript dataset
        :return:
        """

        def critique_outcome(transcript_str, outcome_str):
            outcome_answer = self.pipeline(f"{self.agent_prompts['outcome_critique_prompt_part1']}"
                                           f"{outcome_str}\n"
                                           f"{self.agent_prompts['outcome_critique_prompt_part2']}"
                                           f"{transcript_str}\n\n "
                                           f"CONCLUSION: ")[0]['generated_text']
            match = re.search(r"\bCONCLUSION:\s*(ISSUE RESOLVED|FOLLOW-UP ACTION NEEDED)\b",
                              outcome_answer, re.IGNORECASE)
            if match:
                return match.group(1)
            else:
                return 'UNKNOWN'

        # Use entire call transcript to determine actual outcome of the call, as often times the support representative
        # suggests a follow-up action on the call and the customer simply acknowledges this
        self.transcripts_df['outcome_with_critique'] = (self.transcripts_df
                                                        .apply(lambda row: critique_outcome(row['raw_transcript'],
                                                                                            row['outcome']), axis=1))
        if save_flag:
            self.transcripts_df.to_csv(project_root / self.app_configuration['data_save_path'], index=False)

        return self.transcripts_df
