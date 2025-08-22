import re

# ---- English patterns ----
GENDER_TERMS = [
    r"\b(woman|women|girl|girls|female|she|her|hers|wife|girlfriend|gf|lady|ladies|feminist|feminazi|mother|mom)\b",
    r"\b(lgbt|trans|transgender|lesbian|gay|queer|nonbinary)\b",
]
VIOLENCE_TERMS = [
    r"\b(rape|raped|rapist|sexual\s*assault|assault|molest|molested|harass|harassed|harassment|stalk|stalking)\b",
    r"\b(abuse|abused|abusive|beat|beating|violence|victim|victimized|traffick|acid\s*attack|dowry)\b",
]
GENDERED_SLURS_OR_MISOGYNY = [
    r"\b(bitch|slut|whore|hoe|cunt|skank)\b",
    r"\b(misogyny|misogynist|misogynistic)\b",
]

# ---- Hindi / Hinglish patterns (minimal, extend as needed) ----
HINDI_GENDER = [
    r"\b(aurat|mahila|ladki|ladkiyan|patni|biwi|premika|ladies)\b",
    r"\b(hijra|kinnar|trans|lgbt|lesbian|gay|queer)\b",
]
HINDI_VIOLENCE = [
    r"\b(balatkar|balatkari|chhed(chha(d)?|chaad)|utpidan|hinsa|maar|pitai)\b",
    r"\b(yaun(\s*|i)utpidan|yaunik\s*hamla|acid\s*attack|dowry|dahej)\b",
]
HINDI_GENDERED_ABUSE = [
    r"\b(randi|kamini|bazaru|kutiya)\b",
]

# Compile
RX_GENDER = re.compile("|".join(GENDER_TERMS), re.IGNORECASE)
RX_VIOLENCE = re.compile("|".join(VIOLENCE_TERMS), re.IGNORECASE)
RX_GENDERED_ABUSE = re.compile("|".join(GENDERED_SLURS_OR_MISOGYNY), re.IGNORECASE)

RX_HI_GENDER = re.compile("|".join(HINDI_GENDER), re.IGNORECASE)
RX_HI_VIOLENCE = re.compile("|".join(HINDI_VIOLENCE), re.IGNORECASE)
RX_HI_GENDERED_ABUSE = re.compile("|".join(HINDI_GENDERED_ABUSE), re.IGNORECASE)

def weaklabel_gender_violence(text: str) -> int:
    """1 = gender (+) (violence OR gendered abuse), English or Hindi/Hinglish."""
    if not isinstance(text, str):
        return 0
    has_gender = bool(RX_GENDER.search(text) or RX_HI_GENDER.search(text))
    has_violence = bool(RX_VIOLENCE.search(text) or RX_HI_VIOLENCE.search(text))
    has_gendered_abuse = bool(RX_GENDERED_ABUSE.search(text) or RX_HI_GENDERED_ABUSE.search(text))
    return int(has_gender and (has_violence or has_gendered_abuse))
