"""
Defines the standard questions for each PII type to be used by the 
QaPiiDetector.

Each PII type is mapped to a list of questions. The detector will iterate
through these questions to find the corresponding PII.
"""

PII_QUESTIONS = {
    "PERSON": [
        "What is the person's name?",
        "Who is the individual mentioned?"
    ],
    "LOCATION": [
        "What is the address?",
        "What is the location mentioned?",
        "Where is the place mentioned?"
    ],
    "PHONE_NUMBER": [
        "What is the phone number?",
        "What is the contact number?"
    ],
    "EMAIL_ADDRESS": [
        "What is the email address?"
    ],
    "CREDIT_CARD": [
        "What is the credit card number?"
    ],
    "SSN": [
        "What is the Social Security Number?",
        "What is the SSN?"
    ],
    "DRIVERS_LICENSE": [
        "What is the driver's license number?"
    ]
}
