import re

# Gender markers
GENDER_TERMS = [
    r"\b(woman|women|girl|girls|female|she|her|hers|wife|girlfriend|gf|lady|ladies|feminist|feminazi|mother|mom)\b",
    r"\b(lgbt|trans|transgender|lesbian|gay|queer|nonbinary)\b",
]
# Violence / abuse terms
VIOLENCE_TERMS = [
    r"\b(rape|raped|rapist|sexual\s*assault|assault|molest|molested|harass|harassed|harassment|stalk|stalking)\b",
    r"\b(abuse|abused|abusive|beat|beating|violence|victim|victimized|traffick|acid\s*attack|dowry)\b",
]
# Gendered slurs
GENDERED_SLURS_OR_MISOGYNY = [
    r"\b(bitch|slut|whore|hoe|cunt|skank)\b",
    r"\b(misogyny|misogynist|misogynistic)\b",
]

RX_GENDER = re.compile("|".join(GENDER_TERMS), re.IGNORECASE)
RX_VIOLENCE = re.compile("|".join(VIOLENCE_TERMS), re.IGNORECASE)
RX_GENDERED_ABUSE = re.compile("|".join(GENDERED_SLURS_OR_MISOGYNY), re.IGNORECASE)

def weaklabel_gender_violence(text: str) -> int:
    """1 = likely gender-violence/victimization; 0 = other"""
    if not isinstance(text, str): 
        return 0
    has_gender = bool(RX_GENDER.search(text))
    has_violence = bool(RX_VIOLENCE.search(text))
    has_gendered_abuse = bool(RX_GENDERED_ABUSE.search(text))
    return int(has_gender and (has_violence or has_gendered_abuse))
