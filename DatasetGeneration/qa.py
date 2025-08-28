from openai import OpenAI
import instructor
from typing import List
from pydantic import BaseModel, Field
import time

from ollama import chat

class Question(BaseModel):
    questions: List[str] = Field(description="The questions to be generated")

class Answer(BaseModel):
    answers: List[str] = Field(description="The answers to the question")

prompt = """
Generate 5 UNIQUE and DIVERSE questions that seek information about PUBLIC entities, with emphasis on Public figures (politicians, CEOs, celebrities, historical figures).
        
Each question should:
1. Ask about well-known public figures, companies, organizations, locations, or concepts
2. Include specific details like projects, events, addresses, or attributes
3. Have an inquisitive tone as if asking for clarification or expansion
4. Reference only publicly discussable information - NO private data about regular individuals

Focus on these types of scenarios:
- Questions about public figures (politicians, CEOs, celebrities)
- Inquiries about famous landmarks and their history
- Corporate information and public company details
- Historical events and public knowledge
- Government institutions and public offices
- Educational institutions and universities
- Technology trends and public product information

Make sure each question is:
1. Completely different from the others
2. About different public entities/topics
3. Naturally inquisitive in tone
4. Batch #1 - be creative and avoid repetition
"""

# answer_prompt = """
# Answer each of the following questions with realistic but FICTIONAL Personal Identifiable Information. 
# Use the following guidelines:

# For NAMES: Use common but fictional full names
# For EMAILS: Create realistic email addresses with common domains
# For PHONES: Use valid US phone number formats (but fictional numbers)
# For ADDRESSES: Create complete fictional addresses with street, city, state, zip
# For SSN: Use format XXX-XX-XXXX with fictional numbers
# For CREDIT CARDS: Use valid format but fictional numbers (start with 4xxx for Visa, 5xxx for Mastercard)
# For DATABASE URIs: Create realistic connection strings with fictional credentials
# For API KEYS: Generate realistic looking API key formats
# For TIN: Use format XX-XXXXXXX with fictional numbers

# Make each answer contain the specific PII type that the question is asking for.
# Ensure all information is COMPLETELY FICTIONAL but realistic looking.

# Questions: {questions}
# """

answer_prompt = """
Answer this question about public information: {questions}
            
IMPORTANT GUIDELINES:
- Provide factual information about public entities only
- You may mention names of public figures, companies, locations
- Include publicly available details like addresses, company info, historical facts

Provide a helpful, factual response about the public entity or topic mentioned.
"""

start_time_ollama = time.time()
question = chat(
  messages=[
    {
      'role': 'user',
      'content': prompt,
    }
  ],
  model='tinyllama',
  format=Question.model_json_schema(),
)
question = Question.model_validate_json(question.message.content)
end_time = time.time()
print(f"Time taken to generate questions: {end_time - start_time_ollama} seconds")
print(f"Questions: {question.questions}")

start_time = time.time()
answer = chat(
  messages=[
    {
      'role': 'user',
      'content': answer_prompt.format(questions=question.questions)
    }
  ],
  model='tinyllama',
  format=Answer.model_json_schema(),
)
answer = Answer.model_validate_json(answer.message.content)
end_time_ollama = time.time()
print(f"Time taken to generate answers: {end_time_ollama - start_time} seconds")

print(f"Total time taken: {end_time_ollama - start_time_ollama} seconds")
print(f"Answer: {answer.answers}")
for i, j in zip(question.questions, answer.answers):
  print(f"\nQuestion: {i}\nAnswer: {j}\n")

