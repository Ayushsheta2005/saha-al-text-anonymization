"""
Entity Type Taxonomy & Schema
Defines the complete PII entity type system with descriptions, examples, and display colors.
"""

ENTITY_SCHEMA = {
    "FULLNAME": {
        "description": "Complete person name (first + last, possibly middle)",
        "examples": ["Igorche Ramtin Eshekary", "Allassane Gianuario Künsch"],
        "min_tokens": 2,
        "max_tokens": 6,
        "color": "#FF6B6B",
    },
    "FIRST_NAME": {
        "description": "Given / first name only",
        "examples": ["Brandy", "Shola", "Teslim"],
        "min_tokens": 1,
        "max_tokens": 1,
        "color": "#FF8E8E",
    },
    "LAST_NAME": {
        "description": "Family name / surname only",
        "examples": ["Haroon", "Eshekary", "Zimeri"],
        "min_tokens": 1,
        "max_tokens": 2,
        "color": "#FFA5A5",
    },
    "ID_NUMBER": {
        "description": "Government-issued ID, document number, or alphanumeric code",
        "examples": ["MARGA7101160M9183", "3LPHEOTUUH", "K4VMKGVUXE"],
        "color": "#4ECDC4",
    },
    "PASSPORT": {
        "description": "Passport number",
        "examples": ["AB1234567", "NQ1887589"],
        "color": "#3BAEA0",
    },
    "SSN": {
        "description": "Social Security Number (US format XXX-XX-XXXX)",
        "examples": ["123-45-6789", "401-96-3063"],
        "color": "#2EA495",
    },
    "PHONE": {
        "description": "Phone number in any format",
        "examples": ["805 099 8045", "+80.20-145-6885", "(509).1090420"],
        "color": "#F7DC6F",
    },
    "EMAIL": {
        "description": "Email address",
        "examples": ["marboon@gmail.com", "NB21@outlook.com", "CD@tutanota.com"],
        "color": "#F4D03F",
    },
    "ADDRESS": {
        "description": "Physical street address",
        "examples": ["Bendravedy Mandya Road", "Church Road 92", "County Road 417 850"],
        "color": "#BB8FCE",
    },
    "DATE": {
        "description": "Calendar date in any format",
        "examples": ["4th August 1942", "21/08/1946", "1990-05-05T00:00:00"],
        "color": "#85C1E9",
    },
    "TIME": {
        "description": "Time of day",
        "examples": ["10:17", "02:05:58", "5:52:48 AM", "1:52 PM"],
        "color": "#7FB3D8",
    },
    "LOCATION": {
        "description": "City, town, region, or place name",
        "examples": ["Mississauga", "Townsend", "Gurugram Tehsil Tikli"],
        "color": "#73C6B6",
    },
    "ORGANIZATION": {
        "description": "Company, institution, or organisation name",
        "examples": ["[ORGANISATIONPLACEHOLDER_14]", "United Brewing"],
        "color": "#F0B27A",
    },
    "ACCOUNT_NUMBER": {
        "description": "Bank or financial account number (long digit sequence)",
        "examples": ["644693204822782691", "8264583817285989"],
        "color": "#E8DAEF",
    },
    "CREDIT_CARD": {
        "description": "Credit or debit card number (16 digits)",
        "examples": ["4111 1111 1111 1111"],
        "color": "#D7BDE2",
    },
    "ZIPCODE": {
        "description": "Postal / ZIP code",
        "examples": ["CA7 4AE", "229125", "43431-9599"],
        "color": "#AED6F1",
    },
    "TITLE": {
        "description": "Honorific or title prefix",
        "examples": ["Mr", "Madame", "Mister", "Miss", "Master", "Mayoress"],
        "color": "#D5F5E3",
    },
    "GENDER": {
        "description": "Gender reference used as PII in context",
        "examples": ["Male", "Female", "Non-binary"],
        "color": "#FADBD8",
    },
    "NUMBER": {
        "description": "Contextual numeric PII (age, count used as identifier)",
        "examples": ["28", "13", "57"],
        "color": "#E5E7E9",
    },
    "OTHER_PII": {
        "description": "PII that doesn't fit other categories",
        "examples": [],
        "color": "#D5D8DC",
    },
    "UNKNOWN": {
        "description": "Detected entity but type unclear — needs human review",
        "examples": [],
        "color": "#ABB2B9",
    },
}
