# Project 4 Instructions: News Topic Classification with LoRA

## 1. Source and scope

This file is an English translation and structured restatement of the Project 4 assignment in the GoalEarn task framework.

- **Source workbook:** [Sina Rezaei / GoalEarn / Immig / Bootcamp](https://docs.google.com/spreadsheets/d/1gE5eWWF8Opc3QNSFNKndOc7-LU7XZrlOsMVtn3eAGAY/edit)
- **Source tab:** [Project4 - Marjan Jalali](https://docs.google.com/spreadsheets/d/1gE5eWWF8Opc3QNSFNKndOc7-LU7XZrlOsMVtn3eAGAY/edit?gid=719532720#gid=719532720)
- **Source ranges:** B2:H11 for the task definition, workflow, skills, tools, and resources; B14:H24 for deliverables and progress tracking.
- **Translation date:** November 11, 2026.
- **Scope:** The linked Project 4 tab. Other projects, the program planner, the career plan, and the performance table are outside this file's scope.

The requirements below come from the source assignment. Unspecified decisions are listed in Section 7 rather than filled in with assumed requirements. The completion checklist restates the assignment for tracking.

## 2. Project title and objective

**Title:** Implement and evaluate a news topic classification model by fine-tuning a small language model with LoRA.

Build a model that automatically classifies news text from the **AG News** dataset into four categories:

| Category | Meaning |
| --- | --- |
| World | World news |
| Sports | Sports news |
| Business | Business news |
| Sci/Tech | Science and technology news |

First evaluate a pretrained base model. Then fine-tune it using **LoRA (Low-Rank Adaptation)** and compare the results. Also implement prompt engineering experiments using zero-shot and few-shot approaches, and compare them with the base model and the LoRA-fine-tuned model.

## 3. Required workflow

### Step 1 — Understand the problem and dataset

**Actions**

1. Obtain and inspect the AG News dataset.
2. Understand the four news categories, data structure, and labels.
3. Review example news texts from each category.

**Required knowledge**

- Basic natural language processing (NLP).
- Text classification, datasets, and labels.
- Splitting data into training and test sets.

**Tools:** Python; Jupyter Notebook or Google Colab; `datasets`; `pandas`.

**Resources supplied in the spreadsheet**

- [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news)
- [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets/index)
- [Hugging Face Transformers tutorial repository](https://github.com/SMMousaviSP/huggingface_transformers_tutorial)
- [DataCamp tutorial: Fine-Tuning Qwen3.5 Small](https://www.datacamp.com/tutorial/fine-tuning-qwen3-5-small)
- [YouTube learning resource](https://www.youtube.com/watch?v=F29Y_y7g6IY)

### Step 2 — Select the base model and establish a baseline

**Actions**

1. Select a small pretrained language model suitable for text classification.
2. Load the model and its tokenizer.
3. Run inference before fine-tuning and evaluate its initial performance on dataset examples.
4. Record baseline test-set metrics for comparison with the trained model, as required by the baseline deliverable.

**Required knowledge**

- Tokenizers and inference.
- Converting model outputs into category labels.
- Understanding the limitations of the base model.

**Tools:** `transformers`; `torch`; Hugging Face Hub.

**Resources supplied in the spreadsheet**

- [Hugging Face model catalog](https://huggingface.co/models)
- [Transformers text classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)

### Step 3 — Prepare data for training

**Actions**

1. Prepare the data for the selected model.
2. Map the four news categories to numerical labels.
3. Tokenize the text and handle padding and truncation.
4. Prepare training, validation, and test datasets.
5. Prevent data leakage during preparation.

**Required knowledge**

- Text preprocessing and tokenization.
- Padding and truncation.
- Label mapping.
- Data leakage prevention.

**Tools:** `datasets`; `transformers`; `pandas`.

**Resources supplied in the spreadsheet**

- [Transformers tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [Transformers text classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)

### Step 4 — Fine-tune the model with LoRA

**Actions**

1. Apply LoRA to the base model.
2. Train using the training dataset.
3. Configure the main training parameters and evaluate during training.
4. Save the trained adapter.

**Required knowledge**

- Fine-tuning and parameter-efficient fine-tuning (PEFT).
- The general mechanism of LoRA.
- Epochs, batch size, learning rate, loss, and evaluation during training.

**Tools:** `transformers`; `peft`; `torch`; `accelerate` if needed.

**Resources supplied in the spreadsheet**

- [PEFT quick tour](https://huggingface.co/docs/peft/main/quicktour)
- [Transformers Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer)

### Step 5 — Evaluate and compare results

**Actions**

1. Evaluate the fine-tuned model on the test set.
2. Compare its performance with the base model.
3. Report accuracy, precision, recall, and F1-score.
4. Include a confusion matrix.
5. Examine incorrectly classified examples and analyze model weaknesses.

**Required knowledge**

- Classification evaluation metrics.
- Confusion matrices.
- Error analysis and interpretation of results.

**Tools:** `evaluate`; `scikit-learn`; `pandas`; `matplotlib`.

**Resources supplied in the spreadsheet**

- [Hugging Face Evaluate documentation](https://huggingface.co/docs/evaluate/index)
- [scikit-learn model evaluation documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Step 6 — Save the model and prepare the final report

**Actions**

1. Save the LoRA adapter and the configuration needed to reload it.
2. Reload the trained model for inference.
3. Run it on several new news texts.
4. Prepare a comparative report covering the baseline and the fine-tuned model.
5. Document experiments, settings, and the interpretation of results.

**Required knowledge**

- Saving and loading models.
- Running inference.
- Documenting experiments and recording settings.
- Analyzing results.

**Tools:** `peft`; `transformers`; Jupyter Notebook; Git.

**Resources supplied in the spreadsheet**

- [Transformers model documentation](https://huggingface.co/docs/transformers/main_classes/model)
- [PEFT quick tour](https://huggingface.co/docs/peft/main/quicktour)

### Step 7 — Design and evaluate prompts, then compare with fine-tuning

**Actions**

1. Design and test **at least three different prompts** for classifying news into World, Sports, Business, and Sci/Tech.
2. Implement both **zero-shot** and **few-shot** approaches.
3. Define model instructions and control the output format.
4. Handle invalid responses and labels outside the four permitted categories.
5. Evaluate prompt-based classification using accuracy and F1-score.
6. Compare the prompt-based methods with the base model and the LoRA-fine-tuned model.
7. Analyze errors and summarize the effectiveness of each method.

**Required knowledge**

- Prompt engineering principles.
- Differences between zero-shot and few-shot prompting.
- Instruction design and output-format control.
- Invalid-response handling.
- Evaluation, method comparison, and error analysis.

**Tools:** Python; `transformers`; `torch`; `scikit-learn`; Jupyter Notebook or Google Colab.

**Resource supplied in the spreadsheet**

- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## 4. Expected deliverables

The spreadsheet lists these outputs in order of importance or delivery priority.

| Priority | Deliverable | Required contents |
| --- | --- | --- |
| 1 | Dataset preparation and analysis code | AG News loading; inspection of examples and labels; preparation of training, validation, and test sets; tokenization. |
| 2 | Base-model inference code and initial evaluation report | Selection of a small language model; execution on test data; recorded baseline metrics for comparison with the trained model. |
| 3 | LoRA-fine-tuned model | Implemented fine-tuning process; saved trained adapter; ability to reload it and classify new news texts. |
| 4 | Final evaluation and comparison report | Comparison of the base and fine-tuned models using accuracy, precision, recall, and F1-score; a confusion matrix; analysis of incorrectly predicted examples. |
| 5 | Integrated, executable final code | Organized and documented code that makes data preparation, training, evaluation, and prediction reproducible. |
| 6 | Technical documentation and lessons learned | Selected model; training settings; LoRA parameters; required hardware resources; implementation challenges; experimental results. |
| 7 | Prompt experiment code and comparison | Code for running prompts; a performance comparison table; a short report analyzing errors and the effectiveness of each method. |

The final comparison must cover the base model, the LoRA-fine-tuned model, and the zero-shot and few-shot prompt experiments. Include accuracy and F1-score for the prompt experiments, alongside the broader classification metrics required for the baseline-versus-LoRA evaluation.

## 5. Completion checklist

- [ ] Inspect AG News, its four categories, labels, and representative examples.
- [ ] Select and document the small pretrained base model.
- [ ] Run and record the baseline evaluation before fine-tuning.
- [ ] Prepare training, validation, and test data with numerical labels and tokenization, preventing leakage.
- [ ] Apply LoRA, train the model, and record training settings and LoRA parameters.
- [ ] Save and reload the trained adapter and required configuration.
- [ ] Evaluate the fine-tuned model and compare it with the baseline using accuracy, precision, recall, F1-score, and a confusion matrix.
- [ ] Inspect misclassified examples and explain model weaknesses.
- [ ] Demonstrate predictions on several new news texts.
- [ ] Test at least three different prompts and implement both zero-shot and few-shot approaches.
- [ ] Handle invalid prompt outputs and disallowed labels.
- [ ] Compare prompt-based methods with the baseline and LoRA model, including accuracy, F1-score, and error analysis.
- [ ] Deliver organized, documented, reproducible code.
- [ ] Deliver the comparison report, technical documentation, and lessons learned, including hardware requirements and implementation challenges.

## 6. Schedule and mentor progress tracking

The source provides fields for scheduling and mentor follow-up, but leaves their values blank.

| Field | Source status |
| --- | --- |
| Start date | Not specified |
| End date | Not specified |
| Whether a mentor meeting is needed before or after each deliverable | Not specified |
| Progress toward each expected output | No progress entries provided |
| Knowledge gained from completing each output | No entries provided |
| Actions the mentee must complete before the next meeting | No actions provided |

For progress updates, the source asks the mentee to describe how far each output has been achieved and what they learned while producing it. It also provides space to record actions required before the next mentor meeting.

Workflow rows 8–9 and deliverable rows 8–10 are empty placeholders in the spreadsheet; they add no requirements.

## 7. Decisions not specified in the assignment

The source does not prescribe:

- An exact model name, architecture, parameter count, or model revision.
- Numerical label IDs or a particular label mapping.
- Training/validation/test split proportions or dataset subset sizes.
- Token length, batch size, learning rate, number of epochs, or exact LoRA settings.
- A hardware configuration or compute budget.
- A minimum target accuracy or F1-score.
- A precision, recall, or F1 averaging convention.
- The number of examples in each few-shot prompt or a required distribution of the three or more prompts between zero-shot and few-shot approaches.
- A mandatory project folder structure, notebook format, or report format.

Choose and document these settings during implementation. The listed learning resources are references supplied by the spreadsheet; they do not make a particular model or tutorial approach mandatory. Their contents were not independently reviewed for this translation.

## 8. Additional source framework link

The resource-column heading also links to a separate [skills and tools reference spreadsheet](https://docs.google.com/spreadsheets/d/1zGaDnzMFG5z5p9-uMDKPQjc6oMawZSm4UTAHC7RzNSI/edit?gid=253413737#). This link is preserved for reference; its contents are not incorporated into this instructions file.
