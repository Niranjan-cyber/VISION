import re
from typing import Dict, Tuple

# Standard Indian State / Union Territory Codes
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DN", "DD", "DL",
    "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ",
    "SK", "TN", "TS", "TR", "UP", "UK", "WB", "BH"
}

# Optical Character Recognition confusion mappings
CHAR_TO_DIGIT: Dict[str, str] = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1", "|": "1", "T": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6", "b": "6",
    "B": "8",
    "P": "9", "g": "9",
}

DIGIT_TO_CHAR: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}

# Regex Patterns
# Standard Indian: State(2 letters) + District(1-2 digits) + Series(1-3 letters) + Number(4 digits)
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

# Bharat Series (BH): Year(2 digits) + BH + Number(4 digits) + Letters(1-2)
BHARAT_SERIES_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Generic Alphanumeric plate: 7 to 10 alphanumeric characters
GENERIC_PLATE_REGEX = re.compile(r"^[A-Z0-9]{7,10}$")

# Common OCR misreads where first letter merges with blue IND stripe or plate border
STATE_PREFIX_CORRECTIONS: Dict[str, str] = {
    "HH": "MH", "WH": "MH", "NH": "MH", "KH": "MH", "OH": "MH",
    "6H": "MH", "4H": "MH", "1H": "MH", "VH": "MH", "EH": "MH",
    "0L": "DL", "OL": "DL", "QL": "DL", "1L": "DL",
    "CJ": "GJ", "QJ": "GJ",
    "KR": "HR", "1R": "HR",
}


def strip_non_alphanumeric(text: str) -> str:
    """Removes all spaces, hyphens, colons, dots, and special symbols, converting to uppercase."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def disambiguate_indian_plate(text: str) -> str:
    """
    Applies syntactic knowledge of Indian registration number schema to correct
    frequently confused OCR characters (e.g. '0' vs 'O', '8' vs 'B', '1' vs 'I', state prefixes).
    """
    chars = list(text)
    n = len(chars)

    # State prefix correction if first letter was distorted by the blue IND stripe
    if n >= 4:
        pref = chars[0] + chars[1]
        if pref in STATE_PREFIX_CORRECTIONS:
            corrected_pref = STATE_PREFIX_CORRECTIONS[pref]
            chars[0], chars[1] = corrected_pref[0], corrected_pref[1]
        elif chars[1] == "H" and chars[0] in {"H", "N", "W", "K", "O", "6", "1", "I", "V", "E", "B"}:
            chars[0] = "M"

    # If first letter was single 'M', 'W', 'H', 'N', 'K' followed by 2 digits (e.g. W50U7737 -> MH50U7737)
    if n == 8 and chars[0] in {"M", "W", "H", "N", "K"} and (chars[1].isdigit() or chars[1] in CHAR_TO_DIGIT) and (chars[2].isdigit() or chars[2] in CHAR_TO_DIGIT):
        chars = ["M", "H"] + chars[1:]
        n = len(chars)

    # Case 1: Standard 10-character plate: [AA][00][AA][0000]
    # e.g., MH 12 AB 1234
    if n == 10:
        # Positions 0, 1 must be letters (State code)
        chars[0] = DIGIT_TO_CHAR.get(chars[0], chars[0])
        chars[1] = DIGIT_TO_CHAR.get(chars[1], chars[1])

        # Positions 2, 3 must be digits (RTO code)
        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])

        # Positions 4, 5 must be letters (Series code)
        chars[4] = DIGIT_TO_CHAR.get(chars[4], chars[4])
        chars[5] = DIGIT_TO_CHAR.get(chars[5], chars[5])

        # Positions 6, 7, 8, 9 must be digits (Registration number)
        for i in range(6, 10):
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])

    # Case 2: 9-character plate with 2-letter series: [AA][0][AA][0000] (single digit RTO)
    # or [AA][00][A][0000] (single letter series, e.g. MH 50 U 7737)
    elif n == 9:
        # Positions 0, 1 must be letters
        chars[0] = DIGIT_TO_CHAR.get(chars[0], chars[0])
        chars[1] = DIGIT_TO_CHAR.get(chars[1], chars[1])

        # Positions 2, 3 must be digits (e.g. 50 in MH50U7737)
        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])

        # Position 4 is the series letter (e.g. 'U' in MH50U7737)
        chars[4] = DIGIT_TO_CHAR.get(chars[4], chars[4])

        # Last 4 digits (positions 5, 6, 7, 8)
        for i in range(5, 9):
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])

    return "".join(chars)


def is_valid_indian_plate(text: str) -> bool:
    """Checks whether the text matches Indian Standard or Bharat Series registration syntax."""
    if not text:
        return False
    if INDIAN_PLATE_REGEX.match(text):
        state_prefix = text[:2]
        return state_prefix in INDIAN_STATE_CODES
    if BHARAT_SERIES_REGEX.match(text):
        return True
    return False


def is_valid_generic_plate(text: str) -> bool:
    """Checks whether the text is a plausible alphanumeric license plate."""
    if not text:
        return False
    return bool(GENERIC_PLATE_REGEX.match(text))


def clean_plate_text(raw_text: str) -> Tuple[str, bool, float]:
    """
    Cleans raw OCR output into a standardized, validated license plate string.

    Returns:
        Tuple of:
            cleaned_text (str): Cleaned and disambiguated uppercase plate string
            is_valid (bool): True if matching valid registration syntax
            confidence_multiplier (float): Quality weight (1.0 for valid Indian syntax,
                                           0.7 for valid generic syntax, 0.2 for partial fragments)
    """
    raw_cleaned = strip_non_alphanumeric(raw_text)
    if not raw_cleaned:
        return "", False, 0.0

    # 1. Direct Indian check
    if is_valid_indian_plate(raw_cleaned):
        return raw_cleaned, True, 1.0

    # 2. Attempt Indian disambiguation
    disambiguated = disambiguate_indian_plate(raw_cleaned)
    if is_valid_indian_plate(disambiguated):
        return disambiguated, True, 1.0

    # 3. Check full generic alphanumeric format (7-10 chars)
    if is_valid_generic_plate(disambiguated):
        return disambiguated, True, 0.80

    if is_valid_generic_plate(raw_cleaned):
        return raw_cleaned, True, 0.80

    # 4. Fallback: Return raw cleaned text as partial fragment
    return raw_cleaned, False, 0.20


def format_indian_plate(text: str) -> str:
    """
    Formats a standard Indian plate into human-readable spaced notation.
    e.g. 'MH50U7737' -> 'MH 50 U 7737', 'MH12AB1234' -> 'MH 12 AB 1234'.
    """
    if not text:
        return ""
    clean = strip_non_alphanumeric(text)
    if len(clean) == 9 and clean[:2].isalpha() and clean[2:4].isdigit() and clean[4].isalpha() and clean[5:].isdigit():
        return f"{clean[:2]} {clean[2:4]} {clean[4]} {clean[5:]}"
    elif len(clean) == 10 and clean[:2].isalpha() and clean[2:4].isdigit() and clean[4:6].isalpha() and clean[6:].isdigit():
        return f"{clean[:2]} {clean[2:4]} {clean[4:6]} {clean[6:]}"
    elif len(clean) == 9 and clean[:2].isalpha() and clean[2:3].isdigit() and clean[3:5].isalpha() and clean[5:].isdigit():
        return f"{clean[:2]} {clean[2:3]} {clean[3:5]} {clean[5:]}"
    return clean
