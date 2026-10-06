# Predicting open LLMs' ability to resist jailbreak attempts

Checkpoint 1: problem description, dataset description, stakeholders, KPIs

Ruoyu (Tony) Guo, Yu Xin, Vera Andersson

Fall 2026

## Problem definition

We plan to build machine learning models such as regression, random forest, clustering, and possibly simple deep learning models to predict whether a jailbreak attempt will succeed on LLMs. We test this on four existing LLMs: `gpt-3.5-turbo-1106`, `gpt-4-0125-preview`, `llama-2-7b-chat-hf`, `vicuna-13b-v1.5`, with a large portion of the data for training (and validation) and the rest for testing. This is especially crucial for evaluating the safety of new models and help model developers identify model weaknesses. Note that this project is tested on relatively small models, so it could have trouble predicting the responses of larger size LLMs.

## Dataset description

We download existing jailbreak dataset from the [JailbreakBench's published artifacts](https://github.com/JailbreakBench/artifacts) github repository. The dataset contains jailbreak attempts on the four aforementioned LLMs. Specifically, the dataset contains the category of attacks, prompt, response, numbers of tokens in the prompt and response, jailbroken result (success/fail), and more. We will engineer new features from observing the prompts such as certain word frequencies, embedding of the prompt, and sentiment of the prompt.

This existing dataset contains 886 prompt-response pairs and is obtained through a one-time download. Of the 886 pairs, there are significantly more positive jailbreak attempts (610) than negative (276). Hardware permitting, we would consider using our own prompt to generate data to enlarge this jailbreak dataset.

## KPIs

We plan to use simple accuracy, precision, and recall to evaluate our models' performance. As the dataset has a skew towards successful jailbreaks, we should also use F1 and Matthews correlation coefficient (MCC) to reduce the impact of class imbalance. Potentially, we can analyze the LLM responses of successful attempts by designing a metric such as helpfulness score. However, this addition step is more on evaluating the safety of the models and would be a second objective of the problem.
