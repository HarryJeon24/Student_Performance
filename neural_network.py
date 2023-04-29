import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

# Load the dataset
data = pd.read_csv('data.csv')

# Preprocess the data
# Note: Replace 'text' with the name of the column containing the text data to be fed into BERT
X = data['text']
y = data['correct']

# Tokenize the text data
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def encode_text(text):
    return tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')


X_encoded = [encode_text(text) for text in X]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Perform k-fold cross-validation
k_fold = 10
kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i, (train_index, test_index) in enumerate(kf.split(X_encoded)):
    X_train, X_test = [X_encoded[i] for i in train_index], [X_encoded[i] for i in test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # Fine-tune BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(y)))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        seed=42,
    )


    def compute_metrics(predictions, labels):
        preds = predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return {'accuracy': acc, 'f1_score': f1}


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=list(zip(X_train, y_train)),
        eval_dataset=list(zip(X_test, y_test)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Save the fine-tuned model
    model_output_dir = f'./bert_models/fold_{i + 1}'
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print(f'Fold {i + 1} - Accuracy: {metrics["eval_accuracy"]:.2f}, F1-score: {metrics["eval_f1_score"]:.2f}')