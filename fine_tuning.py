"""Fine-tuning utilities for LLM adaptation."""

import json
from typing import List, Dict, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


class FineTuner:
      """Fine-tuning utilities for adapting LLMs to specific domains."""

    def __init__(self, model_name: str = "gpt2"):
              """Initialize fine-tuner.

                              Args:
                                          model_name: Hugging Face model identifier
                                                  """
              self.model_name = model_name
              self.tokenizer = AutoTokenizer.from_pretrained(model_name)
              self.model = AutoModelForCausalLM.from_pretrained(model_name)
              self.training_history = []

    def prepare_training_data(self, data_path: str) -> Dict:
              """Prepare training data from JSONL file.

                              Args:
                                          data_path: Path to JSONL training data

                                                              Returns:
                                                                          Dictionary with tokenized training data
                                                                                  """
              data = []
              with open(data_path, 'r') as f:
                            for line in f:
                                              data.append(json.loads(line))
                                      return {"data": data}

          def tokenize_data(self, texts: List[str]) -> Dict:
                    """Tokenize training texts.

                                    Args:
                                                texts: List of training texts

                                                                    Returns:
                                                                                Tokenized data dictionary
                                                                                        """
                    encodings = self.tokenizer(
                        texts,
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    return encodings

    def train(self, data_path: str, epochs: int = 3, batch_size: int = 16) -> Dict:
              """Fine-tune model on provided data.

                              Args:
                                          data_path: Path to training data
                                                      epochs: Number of training epochs
                                                                  batch_size: Batch size for training

                                                                                      Returns:
                                                                                                  Training history
                                                                                                          """
              training_data = self.prepare_training_data(data_path)
              texts = [item.get('text', '') for item in training_data['data']]

        training_args = TrainingArguments(
                      output_dir="./fine_tuned_model",
                      num_train_epochs=epochs,
                      per_device_train_batch_size=batch_size,
                      save_steps=100,
                      save_total_limit=2,
        )

        trainer = Trainer(
                      model=self.model,
                      args=training_args,
                      train_dataset=texts[:10],
        )

        result = trainer.train()
        self.training_history.append({
                      "timestamp": datetime.now().isoformat(),
                      "model": self.model_name,
                      "epochs": epochs,
                      "loss": float(result.training_loss) if result.training_loss else None
        })
        return self.training_history[-1]


if __name__ == "__main__":
      tuner = FineTuner()
      result = tuner.train("data.jsonl")
      print(f"Training complete: {result}")
