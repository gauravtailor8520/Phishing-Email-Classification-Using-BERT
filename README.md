# Email Spam Classification: Detecting Spam Emails

## Overview

This project focuses on detecting spam emails using advanced Natural Language Processing (NLP) techniques. We employ BERT-based models, specifically DistilBERT and TinyBERT, to classify emails as spam or legitimate. The project includes data preprocessing, model training, evaluation, and compression techniques to enhance efficiency.

## Video Demo

https://github.com/user-attachments/assets/c2d747ca-9d8e-4606-901a-fb12c7ee6fea

## Table of Contents

1. [Introduction](#introduction)
2. [Methods](#methods)
    - [Dataset Overview](#dataset-overview)
    - [Pre-trained Transformer Models](#pre-trained-transformer-models)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Fine-tuning](#model-fine-tuning)
    - [Implementation Details](#implementation-details)
    - [Model Compression](#model-compression)
3. [Results and Discussion](#results-and-discussion)
    - [TinyBERT](#tinybert)
    - [DistilBERT](#distilbert)
    - [Model Compression](#model-compression-results)
4. [Conclusion](#conclusion)
5. [References](#references)
6. [Appendices](#appendices)
7. [Setup Instructions](#setup-instructions)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

Emails are integral to daily communication, making them prime targets for phishing attacks. Phishing emails deceive recipients into revealing sensitive information, leading to financial losses and data breaches. Traditional detection methods struggle with the evolving tactics of attackers, necessitating more sophisticated approaches.

## Methods

### Dataset Overview

The dataset was compiled by researchers to study phishing email tactics. It combines emails from a variety of sources to create a comprehensive resource for analysis. This dataset contains approximately 82,500 emails where there are 42,891 spam emails and 39,595 legitimate emails.

### Pre-trained Transformer Models

To address the email classification task, we employed two transformer-based models: DistilBERT and TinyBERT, each offering distinct advantages in performance and computational efficiency:

- **DistilBERT**: A streamlined version of BERT, DistilBERT retains 97% of BERT's language understanding while being 60% faster and requiring 40% less memory.
- **TinyBERT**: Even more compact, TinyBERT is specifically tailored for resource-constrained environments. Despite its reduced size, it effectively maintains much of BERT's original performance.

### Data Preprocessing

The email data was preprocessed to ensure compatibility with the transformer models. The preprocessing steps included:

- **Tokenization**: We utilized the respective tokenizers for DistilBERT and TinyBERT to convert email texts into tokenized inputs that the models can process.
- **Text Cleaning**: The email texts were cleaned by converting them to lowercase, removing stop words, and performing stemming.

### Model Fine-tuning

Each model was fine-tuned on the email classification task using a supervised learning approach:

- **Training**: The models were fine-tuned on the labeled email dataset, with a training procedure that included the optimization of the cross-entropy loss function.
- **Evaluation**: We evaluated the performance of the models on a held-out test set using metrics such as precision, recall, and F1-score.

### Implementation Details

- **Hardware and Software**: The experiments were conducted using a standard computing environment with GPU and TPU support to expedite training and evaluation. The models were implemented using the Hugging Face Transformers library and PyTorch.
- **Hyperparameters**: The models were trained with a learning rate of 2e-5 and a batch size of 16. The learning rate was adjusted using a linear scheduler with warm-up steps.

### Model Compression

Model compression techniques such as pruning, quantization, and knowledge distillation are key strategies for making large deep learning models more efficient.

## Results and Discussion

We trained TinyBERT and DistilBERT models to classify emails as either spam or not spam. The training process involved fine-tuning the pre-trained models on our labeled email dataset. The dataset was split in an 80/20 ratio into training and test (validation) sets to monitor the models' performance and prevent overfitting.

### TinyBERT

- **Training and Validation Loss**: The training and validation loss curves show a consistent decrease over the epochs, indicating that the model is learning effectively.
- **Classification Performance**: The model achieved a precision, recall, and F1-score of 0.99 for both ‘Not Spam’ and ‘Spam’ classes, with support of 16498 instances.

### DistilBERT

- **Training and Validation Loss**: The training and validation loss curves for DistilBERT show a similar trend to those of the BERT model.
- **Classification Performance**: The classification report for DistilBERT shows high performance metrics, with precision, recall, and F1-score all close to 0.99 for both ‘Not Spam’ and ‘Spam’ classes.

### Model Compression Results

Three compressions were done for the DistilBERT model: Pruning, Knowledge Distillation, and Quantization. The performance results are shown in the following table:

| Compression Technique | Training Accuracy | Training F1 Score | Training Loss | Test Accuracy | Test F1 Score | Test Loss | Epochs |
|-----------------------|-------------------|-------------------|---------------|---------------|---------------|-----------|--------|
| Pruned                | 0.9988            | 0.9988            | 0.0041        | 0.9930        | 0.9930        | 0.0278    | 7      |
| Knowledge Distillation| -                 | -                 | -             | 0.9869        | 0.9869        | 0.0453    | 1      |
| Quantization          | 0.9898            | 0.9898            | 0.0442        | 0.9796        | 0.9796        | -         | 1      |

## Conclusion

In this project, we classified emails into spam and non-spam categories using TinyBERT, DistilBERT, and compressed variants of DistilBERT. Our results indicate that TinyBERT excels in identifying spam emails, leading to a lower rate of false negatives. Conversely, DistilBERT is more effective at classifying non-spam emails, resulting in fewer false positives. Among the compressed models, the pruned DistilBERT demonstrated the highest performance.

## References

1. Al-Subaiey, A., Antora, K. F., Al-Thani, M., Khandakar, A., Alam, N. A., & Zaman, S. A. U. (Year). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. 2405.11619 (arxiv.org)
2. Lee, Y., & Saxe, J. (2020). CATBERT: Context-Aware Tiny BERT for Detecting Social Engineering Emails (A Preprint). arXiv. https://arxiv.org/abs/2010.03484.
3. Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619
4. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv. https://arxiv.org/abs/1910.01108
5. Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). TinyBERT: Distilling BERT for Natural Language Understanding. arXiv. https://arxiv.org/abs/1909.10351

## Appendices

- **Figure 1**: TinyBERT Training and Validation Loss over Epochs
- **Figure 2**: TinyBERT Classification Report
- **Figure 3**: DistilBERT Training and Validation Loss over Epochs
- **Figure 4**: DistilBERT Classification Report


## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/email-spam-classification.git
   cd email-spam-classification
