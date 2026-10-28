"""Task 4: Fine-tune DistilBERT on Task 3 CSV using Hugging Face Trainer.
Run after Task 3 has created ../artifacts/customer_feedback.csv, or pass --data."""
import argparse, os, time
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

MODEL_NAME = "distilbert/distilbert-base-uncased"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data", default="../artifacts/customer_feedback.csv")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_rows", type=int, default=600, help="Use a subset for faster training; 0 means all rows")
    parser.add_argument("--output", default="saved_model")
    args=parser.parse_args()
    df=pd.read_csv(args.data).dropna(subset=["text","label"])
    if args.max_rows and len(df)>args.max_rows:
        df=df.groupby("label", group_keys=False).apply(
            lambda group: group.sample(
                n=min(len(group), max(1, args.max_rows // df["label"].nunique())),
                random_state=42
            )
        ).reset_index(drop=True)
    encoder=LabelEncoder(); df["labels"]=encoder.fit_transform(df["label"].astype(str))
    label_names=list(encoder.classes_)
    train_df, test_df=train_test_split(df, test_size=.2, random_state=42, stratify=df["labels"])
    train_df, val_df=train_test_split(train_df, test_size=.15, random_state=42, stratify=train_df["labels"])
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)
    def to_ds(frame):
        ds=Dataset.from_pandas(frame[["text","labels"]].reset_index(drop=True))
        return ds.map(tokenize, batched=True)
    train_ds, val_ds, test_ds=map(to_ds,(train_df,val_df,test_df))
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_names),
        id2label={i:label for i,label in enumerate(label_names)},
        label2id={label:i for i,label in enumerate(label_names)})
    def metrics(eval_pred):
        logits, labels=eval_pred
        preds=np.argmax(logits,axis=-1)
        return {"accuracy":accuracy_score(labels,preds),"f1":f1_score(labels,preds,average="macro")}
    train_args=TrainingArguments(
        output_dir="training_checkpoints", num_train_epochs=args.epochs,
        learning_rate=2e-5, per_device_train_batch_size=8,
        per_device_eval_batch_size=16, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=10, report_to="none",
        fp16=bool(__import__("torch").cuda.is_available()),
        save_total_limit=2, seed=42)
    trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=metrics
)
    start=time.perf_counter(); trainer.train(); training_time=time.perf_counter()-start
    result=trainer.evaluate(test_ds)
    os.makedirs(args.output,exist_ok=True)
    trainer.save_model(args.output); tokenizer.save_pretrained(args.output)
    with open(os.path.join(args.output,"labels.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(label_names))
    pred_start=time.perf_counter(); trainer.predict(test_ds); inference_time=(time.perf_counter()-pred_start)/len(test_ds)
    print("Label names:", label_names)
    print("Test metrics:", result)
    print(f"Fine-tuning time: {training_time:.2f}s; approximate inference time per row: {inference_time:.4f}s")
    pd.DataFrame([{"model":"DistilBERT fine-tuned","accuracy":result.get("eval_accuracy"),
      "macro_f1":result.get("eval_f1"),"training_time_seconds":training_time,
      "inference_time_seconds_per_text":inference_time}]).to_csv(os.path.join(args.output,"transformer_metrics.csv"),index=False)

if __name__=="__main__": main()
