"""Advanced prompt engineering techniques for LLM optimization."""

from typing import List, Dict, Optional
from enum import Enum
import json


class PromptTemplate(Enum):
      """Common prompt engineering templates."""
      ZERO_SHOT = "zero_shot"
      FEW_SHOT = "few_shot"
      CHAIN_OF_THOUGHT = "chain_of_thought"
      ROLE_PLAY = "role_play"


class PromptEngineer:
      """Advanced prompt optimization framework."""

    def __init__(self):
              """Initialize prompt engineer."""
              self.prompt_history = []
              self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
              """Load prompt templates.

                              Returns:
                                          Dictionary of prompt templates
                                                  """
              return {
                  "zero_shot": "Answer the following question: {question}",
                  "few_shot": "Examples:\n{examples}\n\nNow answer: {question}",
                  "cot": "Let's think step by step. {question}\nStep 1:",
                  "role": "As a {role}, respond to: {question}"
              }

    def create_zero_shot(self, question: str) -> str:
              """Create zero-shot prompt.

                              Args:
                                          question: User question

                                                              Returns:
                                                                          Formatted zero-shot prompt
                                                                                  """
              prompt = self.templates["zero_shot"].format(question=question)
              return self._log_prompt(prompt, PromptTemplate.ZERO_SHOT)

    def create_few_shot(
              self,
              examples: List[Dict[str, str]],
              question: str
    ) -> str:
              """Create few-shot prompt.

                              Args:
                                          examples: List of example input-output pairs
                                                      question: User question

                                                                          Returns:
                                                                                      Formatted few-shot prompt
                                                                                              """
              example_text = "\n".join(
                  f"Q: {e['input']}\nA: {e['output']}" for e in examples
              )
              prompt = self.templates["few_shot"].format(
                  examples=example_text,
                  question=question
              )
              return self._log_prompt(prompt, PromptTemplate.FEW_SHOT)

    def create_chain_of_thought(self, question: str) -> str:
              """Create chain-of-thought prompt.

                              Args:
                                          question: Complex question requiring reasoning

                                                              Returns:
                                                                          Formatted chain-of-thought prompt
                                                                                  """
              prompt = self.templates["cot"].format(question=question)
              return self._log_prompt(prompt, PromptTemplate.CHAIN_OF_THOUGHT)

    def create_role_play(self, role: str, question: str) -> str:
              """Create role-play prompt.

                              Args:
                                          role: Role for LLM to assume
                                                      question: Question for the role

                                                                          Returns:
                                                                                      Formatted role-play prompt
                                                                                              """
              prompt = self.templates["role"].format(
                  role=role,
                  question=question
              )
              return self._log_prompt(prompt, PromptTemplate.ROLE_PLAY)

    def _log_prompt(self, prompt: str, template: PromptTemplate) -> str:
              """Log prompt to history.

                              Args:
                                          prompt: The prompt text
                                                      template: Template type used

                                                                          Returns:
                                                                                      The prompt
                                                                                              """
              self.prompt_history.append({
                  "prompt": prompt,
                  "template": template.value
              })
              return prompt


if __name__ == "__main__":
      engineer = PromptEngineer()
      prompt = engineer.create_chain_of_thought(
          "What is the sum of 15 and 23?"
      )
      print(prompt)
