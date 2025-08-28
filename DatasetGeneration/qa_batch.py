# Modified data_generation.py for Public Entity Q&A Generation

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

class PublicEntityQAGenerator:
    def __init__(self, model='qwen2:1.5b', batch_size=10):
        self.model = model
        self.batch_size = batch_size
        self.used_questions = set()
        self.used_answers = set()
        self.client = AsyncClient()
        
        # Categories for public entity questions
        self.entity_categories = [
            "Public figures (politicians, CEOs, celebrities, historical figures)",
            "Famous landmarks and monuments (Eiffel Tower, Statue of Liberty, etc.)",
            "Corporate headquarters and public company information",
            "Historical events and public knowledge",
            "Government institutions and public offices",
            "Educational institutions and universities",
            "Technology companies and their products",
            "Cultural institutions (museums, theaters, public venues)",
            "Sports teams, athletes, and public sporting events",
            "News-worthy topics and current public discussions"
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
        """Generate a batch of unique public entity questions"""
        # Vary the focus category to encourage diversity
        category_focus = self.entity_categories[batch_num % len(self.entity_categories)]
        
        prompt = f"""
        Generate {self.batch_size} UNIQUE and DIVERSE questions that seek information about PUBLIC entities, with emphasis on {category_focus}.
        
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
        4. Batch #{batch_num} - be creative and avoid repetition
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat(
                    messages=[{'role': 'user', 'content': prompt}],
                    model='tinyllama',
                    format=Question.model_json_schema(),
                    options={
                        'temperature': 0.8 + (attempt * 0.1),
                        'num_ctx': 4096,
                        'num_predict': 800,
                        'num_gpu': -1,
                    }
                )
                
                questions_obj = Question.model_validate_json(response.message.content)
                
                # Filter out duplicates
                unique_questions = []
                for q in questions_obj.questions:
                    if not self._is_duplicate_question(q) and len(q.strip()) > 15:
                        unique_questions.append(q.strip())
                
                if len(unique_questions) >= self.batch_size * 0.7:
                    return unique_questions[:self.batch_size]
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback questions
        return [f"What do you know about {category_focus.split()[0].lower()} in the public domain?"]
    
    async def generate_answers_batch(self, questions: List[str]) -> List[str]:
        """Generate answers that mention entities but contain NO private PII"""
        tasks = []
        for question in questions:
            # Specialized prompt for public entity answers
            answer_prompt = f"""
            Answer this question about public information: "{question}"
            
            IMPORTANT GUIDELINES:
            - Provide factual information about public entities only
            - You may mention names of public figures, companies, locations
            - Include publicly available details like addresses, company info, historical facts
                       
            Provide a helpful, factual response about the public entity or topic mentioned.
            """
            
            task = self.client.chat(
                messages=[{
                    'role': 'user',
                    'content': answer_prompt
                }],
                model='qwen2:1.5b',
                options={
                    'temperature': 0.7,
                    'num_ctx': 2048,
                    'num_predict': 200,
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
                    answers.append(f"Alternative public information about: {questions[i][:50]}...")
        
        return answers
    
    async def generate_qa_dataset(self, total_pairs: int = 5000) -> List[dict]:
        """Generate the complete public entity Q&A dataset"""
        print(f"Generating {total_pairs} Public Entity Q&A pairs in batches of {self.batch_size}...")
        
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
                        "batch": batch_num + 1,
                    })
            
            batch_time = time.time() - batch_start
            progress = (batch_num + 1) / total_batches * 100
            estimated_total = (time.time() - start_time) / (batch_num + 1) * total_batches
            
            print(f"Batch {batch_num + 1}/{total_batches} ({progress:.1f}%) - "
                  f"{len(questions)} Q&As in {batch_time:.2f}s - "
                  f"ETA: {estimated_total - (time.time() - start_time):.0f}s")
            
            # Save checkpoint every 20 batches
            if (batch_num + 1) % 50 == 0:
                self.save_dataset(dataset, f"public_entity_qa_checkpoint_{batch_num + 1}.json")
        
        total_time = time.time() - start_time
        print(f"\nCompleted! Generated {len(dataset)} Public Entity Q&A pairs in {total_time:.2f}s")
        print(f"Average: {total_time/len(dataset):.3f}s per Q&A pair")
        print(f"Unique questions: {len(self.used_questions)}")
        print(f"Unique answers: {len(self.used_answers)}")
        
        return dataset[:total_pairs]
    
    def save_dataset(self, dataset: List[dict], filename: str = "public_entity_qa_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filename}")

# Usage
async def main():
    generator = PublicEntityQAGenerator(model='qwen2:1.5b', batch_size=10)
    dataset = await generator.generate_qa_dataset(1000)
    generator.save_dataset(dataset)
    
    # Display sample
    print("\nSample Public Entity Q&A pairs:")
    for i in range(min(5, len(dataset))):
        print(f"\nQ{i+1}: {dataset[i]['question']}")
        print(f"A{i+1}: {dataset[i]['answer']}")
        print(f"Category: {dataset[i]['category']}")

if __name__ == "__main__":
    asyncio.run(main())