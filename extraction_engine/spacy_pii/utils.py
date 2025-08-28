"""
Utility functions for spaCy PII detection
"""

import json
import logging
import os
from datetime import datetime


def setup_logging(log_prefix="spacy_pii_detection"):
    """Set up logging configuration for both console and file output."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/{log_prefix}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries."""
    text_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            text_list.append({
                "id": data.get('id'), 
                "text": data.get('text'), 
                "pii": data.get('pii_types', [])
            })
    return text_list


def normalize_entity_type(entity_type: str) -> str:
    """Normalize entity types to consistent format."""
    entity_mapping = {
        'NAME': 'name',
        'EMAIL': 'email', 
        'CREDIT_CARD': 'credit_card',
        'DB_URI': 'db_uri',
        'API_KEY': 'api_key',
        'PHONE_NUMBER': 'phone',
        'ADDRESS': 'address'
    }
    return entity_mapping.get(entity_type.upper(), entity_type.lower())


def deduplicate_entities(entities):
    """
    Remove overlapping entities, keeping the longest/most specific match.
    
    Args:
        entities: List of entity dictionaries with start/end positions
        
    Returns:
        List of deduplicated entities
    """
    if not entities:
        return []
    
    unique_entities = []
    seen_ranges = set()
    
    # Sort by start index to prioritize longer matches if they start at the same time
    entities.sort(key=lambda p: (p['start'], -p['end']))
    
    for entity in entities:
        # Check if this range overlaps with an already added entity
        if not any(entity['start'] < r[1] and entity['end'] > r[0] for r in seen_ranges):
            unique_entities.append(entity)
            seen_ranges.add((entity['start'], entity['end']))
    
    return unique_entities
