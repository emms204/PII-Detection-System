from typing import List, Set
from pydantic import BaseModel, Field
import time
import json
import asyncio
from ollama import AsyncClient, chat
import hashlib

class Question(BaseModel):
    questions: List[str] = Field(description="The questions to be generated")

class Answer(BaseModel):
    answers: List[str] = Field(description="The answers to the question")

class QABatchGenerator:
    def __init__(self, model='qwen2:1.5b', batch_size=10):
        self.model = model
        self.batch_size = batch_size
        self.used_questions = set()
        self.used_answers = set()
        self.client = AsyncClient()
        
        self.categories = [
            "Business or corporate announcements",
            "Technical documentation or instructions from Developers/Engineers/IT Experts/Tech Gurus", 
            "Day-to-Day Conversations from every field e.g Government, Finance, Health, Politics, Agriculture, Education, Sports, Entertainment",
            "General product descriptions from every field e.g Government, Finance, Health, Politics, Agriculture, Education, Sports, Entertainment",
            "Scientific statements",
            "General statements"
        ]
    
    def _hash_text(self, text: str) -> str:
        """Create a hash for duplicate detection"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def _is_duplicate_question(self, question: str) -> bool:
        """Check if question is duplicate"""
        q_hash = self._hash_text(question)
        if q_hash in self.used_questions:
            return True
        self.used_questions.add(q_hash)
        return False
    
    def _is_duplicate_answer(self, answer: str) -> bool:
        """Check if answer is duplicate"""
        a_hash = self._hash_text(answer)
        if a_hash in self.used_answers:
            return True
        self.used_answers.add(a_hash)
        return False
    
    async def generate_question_batch(self, batch_num: int) -> List[str]:
        """Generate a batch of unique questions"""
        # Vary the prompt to encourage diversity
        category_focus = self.categories[batch_num % len(self.categories)]
        
        prompt = f"""
        Generate {self.batch_size} UNIQUE and DIVERSE questions, with emphasis on {category_focus}.
        Also include questions from other categories:
        - Business or corporate announcements
        - Technical documentation or instructions
        - Day-to-Day Conversations from various fields
        - General product descriptions
        - Scientific statements
        - General statements
        
        Make sure each question is:
        1. Completely different from the others
        2. Specific and detailed
        3. From different domains/contexts
        4. Batch #{batch_num} - be creative and avoid repetition
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat(
                    messages=[{'role': 'user', 'content': prompt}],
                    model=self.model,
                    format=Question.model_json_schema(),
                    options={
                        'temperature': 0.8 + (attempt * 0.1),  # Increase randomness on retries
                        'num_ctx': 4096,
                        'num_predict': 800,
                        'num_gpu': -1,
                    }
                )
                
                questions_obj = Question.model_validate_json(response.message.content)
                
                # Filter out duplicates
                unique_questions = []
                for q in questions_obj.questions:
                    if not self._is_duplicate_question(q) and len(q.strip()) > 10:
                        unique_questions.append(q.strip())
                
                if len(unique_questions) >= self.batch_size * 0.7:  # Accept if we get 70% unique
                    return unique_questions[:self.batch_size]
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback: generate simple questions if all retries fail
        return [f"What is the purpose of {category_focus.split()[0].lower()} in batch {batch_num}?"]
    
    async def generate_answers_batch(self, questions: List[str]) -> List[str]:
        """Generate answers for a batch of questions"""
        # Process answers concurrently for speed
        tasks = []
        for i, question in enumerate(questions):
            task = self.client.chat(
                messages=[{
                    'role': 'user',
                    'content': f"Answer this question briefly and specifically (2-3 sentences max): {question}"
                }],
                model=self.model,
                options={
                    'temperature': 0.7,
                    'num_ctx': 2048,
                    'num_predict': 150,
                    'num_gpu': -1,
                }
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        answers = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                answers.append(f"Error generating answer for question {i+1}")
            else:
                answer = response.message.content.strip()
                if not self._is_duplicate_answer(answer):
                    answers.append(answer)
                else:
                    # Generate alternative if duplicate
                    answers.append(f"Alternative answer for: {questions[i][:50]}...")
        
        return answers
    
    async def generate_qa_dataset(self, total_pairs: int = 5000) -> List[dict]:
        """Generate the complete Q&A dataset"""
        print(f"Generating {total_pairs} Q&A pairs in batches of {self.batch_size}...")
        
        dataset = []
        total_batches = (total_pairs + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        
        for batch_num in range(total_batches):
            batch_start = time.time()
            
            # Generate questions for this batch
            questions = await self.generate_question_batch(batch_num)
            
            # Generate answers for these questions
            answers = await self.generate_answers_batch(questions)
            
            # Combine into Q&A pairs
            for q, a in zip(questions, answers):
                if len(dataset) < total_pairs:
                    dataset.append({
                        "id": len(dataset) + 1,
                        "question": q,
                        "answer": a,
                        "batch": batch_num + 1
                    })
            
            batch_time = time.time() - batch_start
            progress = (batch_num + 1) / total_batches * 100
            estimated_total = (time.time() - start_time) / (batch_num + 1) * total_batches
            
            print(f"Batch {batch_num + 1}/{total_batches} ({progress:.1f}%) - "
                  f"{len(questions)} Q&As in {batch_time:.2f}s - "
                  f"ETA: {estimated_total - (time.time() - start_time):.0f}s")
            
            # Save checkpoint every 50 batches
            if (batch_num + 1) % 50 == 0:
                self.save_dataset(dataset, f"qa_dataset_checkpoint_{batch_num + 1}.json")
        
        total_time = time.time() - start_time
        print(f"\nCompleted! Generated {len(dataset)} Q&A pairs in {total_time:.2f}s")
        print(f"Average: {total_time/len(dataset):.3f}s per Q&A pair")
        print(f"Unique questions: {len(self.used_questions)}")
        print(f"Unique answers: {len(self.used_answers)}")
        
        return dataset[:total_pairs]
    
    def save_dataset(self, dataset: List[dict], filename: str = "qa_dataset_5000.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filename}")

# Usage
async def main():
    generator = QABatchGenerator(model='qwen2:1.5b', batch_size=10)
    dataset = await generator.generate_qa_dataset(5000)
    generator.save_dataset(dataset)
    
    # Display sample
    print("\nSample Q&A pairs:")
    for i in range(min(3, len(dataset))):
        print(f"\nQ{i+1}: {dataset[i]['question']}")
        print(f"A{i+1}: {dataset[i]['answer']}")

if __name__ == "__main__":
    asyncio.run(main())