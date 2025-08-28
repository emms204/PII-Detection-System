"""
Validation functions for PII entities
"""

import re
import math
import phonenumbers
from typing import Union


def validate_credit_card_luhn(card_number: str) -> bool:
    """
    Validates a credit card number using the Luhn algorithm.
    
    Args:
        card_number: Credit card number string (may contain spaces/dashes)
        
    Returns:
        True if valid according to Luhn algorithm, False otherwise
    """
    # Remove spaces and dashes
    clean_number = re.sub(r'[- ]', '', card_number)
    
    # Check if all digits
    if not clean_number.isdigit():
        return False
        
    try:
        digits = [int(d) for d in clean_number]
        checksum = sum(digits[-1::-2]) + sum(sum(divmod(d * 2, 10)) for d in digits[-2::-2])
        return checksum % 10 == 0
    except (ValueError, IndexError):
        return False


def calculate_shannon_entropy(s: str) -> float:
    """
    Calculates the Shannon entropy for a string to detect randomness.
    
    Args:
        s: Input string
        
    Returns:
        Shannon entropy value (higher = more random)
    """
    if not s:
        return 0
    entropy = 0
    for x in set(s):
        p_x = float(s.count(x)) / len(s)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy


def validate_email_format(email: str) -> bool:
    """
    Validates email address format using regex.
    
    Args:
        email: Email address string
        
    Returns:
        True if valid email format, False otherwise
    """
    email_regex = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
    return bool(email_regex.match(email.strip()))


def validate_phone_number(phone: str, region: str = "US") -> bool:
    """
    Validates phone number using phonenumbers library.
    
    Args:
        phone: Phone number string
        region: Region code (default: US)
        
    Returns:
        True if valid phone number, False otherwise
    """
    try:
        parsed_number = phonenumbers.parse(phone, region)
        return phonenumbers.is_valid_number(parsed_number)
    except phonenumbers.NumberParseException:
        return False


def validate_db_uri(uri: str) -> bool:
    """
    Validates database URI format using regex.
    
    Args:
        uri: Database URI string
        
    Returns:
        True if valid DB URI format, False otherwise
    """
    db_uri_regex = re.compile(
        r'\b(?:mysql|postgres|mongodb(?:\+srv)?)://[\w.-]+:[\w.-]+@[\w.-]+\b', 
        re.IGNORECASE
    )
    return bool(db_uri_regex.match(uri.strip()))


def validate_api_key_entropy(api_key: str, min_entropy: float = 3.5) -> bool:
    """
    Validates API key using entropy analysis and known prefixes.
    
    Args:
        api_key: API key string
        min_entropy: Minimum entropy threshold
        
    Returns:
        True if likely a valid API key, False otherwise
    """
    # Known API key prefixes
    known_prefixes = [
        'sk_live_', 'pk_live_', 'rk_live_', 'sk_test_', 'pk_test_', 
        'rk_test_', 'AIza', 'ghp_', 'glpat-'
    ]
    
    # Check for known prefixes
    has_known_prefix = any(api_key.startswith(prefix) for prefix in known_prefixes)
    
    # Check entropy
    entropy = calculate_shannon_entropy(api_key)
    has_high_entropy = entropy > min_entropy
    
    # Check minimum length
    has_min_length = len(api_key) >= 11
    
    # Valid if it has known prefix OR (high entropy AND min length)
    return has_known_prefix or (has_high_entropy and has_min_length)


def validate_pii_entity(entity_type: str, value: str) -> bool:
    """
    Validates a PII entity based on its type.
    
    Args:
        entity_type: Type of PII entity
        value: Value to validate
        
    Returns:
        True if validation passes, False otherwise
    """
    validators = {
        'credit_card': validate_credit_card_luhn,
        'email': validate_email_format,
        'phone': validate_phone_number,
        'phone_number': validate_phone_number,
        'db_uri': validate_db_uri,
        'api_key': validate_api_key_entropy
    }
    
    validator = validators.get(entity_type.lower())
    if validator:
        try:
            return validator(value)
        except Exception:
            return False
    
    # For types without specific validators, assume valid
    return True
