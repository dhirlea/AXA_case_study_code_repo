# Call Transcripts Analysis Case Study 

## 1. Getting Started

### 1.1 Build Project Dependencies
From the root directory of the code repository, run the following commands in a terminal window.

```
conda env create -f requirements.yaml
conda activate transcripts_env
``` 

### 1.2 Enable Llama LLM Model 
LLM setup using HuggingFace - need to be permissioned for Llama 3.2 class of models on the HuggingFace Hub (e.g. https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). Once permission is granted by META (may take some time, 1h in my case), set up a user token on the HuggingFace Hub (https://huggingface.co/docs/hub/en/security-tokens). Make sure the token has *read access to contents of all public gated repos you can access* when editing access token permissions.

In a terminal window, run ```huggingface-cli login``` and enter the token above when prompted. When asked *Add token as git credential?*, select *n*. Once this step is completed, access to the Llama 3.2 model family becomes available within the application.

### 1.3 Run The Application Locally

With the *transcripts_env* active, run the following command in the terminal to run all application modules.

``` 
python SRC/app.py --run_all
``` 

Individual modules such as classification, critique and evaluation can be run separately, by including only the relevant flag from the following list [--run_classification, --run_critique, --run_evaluation]. For example, to run evaluation:

``` 
python SRC/app.py --run_evaluation
``` 

If running from a PyCharm IDE instead of the CLI, modify the run configuration of *app.py* to the root directory of the project, namely to *call_transcripts_project*, and mark *SRC* as the Sources Root of the project. 

### 1.3 Run Testing Suite
From the root directory of the code repository, run the following.
```
pytest
``` 

## 2. Check Results
Prediction results are saved under *data/transcripts_predictions.csv* and the insight visuals are saved under *data/exhibits*.






