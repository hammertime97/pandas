"""Word lists and patterns that drive the text side of virality scoring.

These are deliberately kept in one place and as plain data so they can be
tuned, extended for a niche, or swapped for another language without touching
the scoring logic.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Pattern, Tuple

# ---------------------------------------------------------------------------
# Hook patterns - what makes someone not swipe in the first two seconds.
# Each pattern carries a weight; a candidate's hook score saturates so that
# stacking five weak patterns never beats one very strong opener.
# ---------------------------------------------------------------------------

# Speech-to-text output is inconsistent about contractions: the same sentence
# comes back as "here's" from one model and "here is" from another. These
# fragments match both, so a hook is never missed on a spelling difference.
HERES = r"here(?:'?s| is)"
THATS = r"that(?:'?s| is)"
ITS = r"it(?:'?s| is)"
YOURE = r"you(?:'?re| are)"
DONT = r"do(?:n'?t| not)"

HOOK_PATTERNS: List[Tuple[Pattern[str], float]] = [
    (re.compile(r"\b(nobody|no one|nobody ever|most people|everyone) (tells|talks|knows|realis|realiz|thinks|does|gets)"), 1.0),
    (re.compile(rf"\b{HERES} (why|how|what|the (reason|thing|truth|problem))\b"), 0.95),
    (re.compile(r"\bthe (biggest|number one|#1|worst|best|hardest|only|real) \w+"), 0.9),
    (re.compile(r"\b\d+\s+(things|ways|reasons|steps|rules|mistakes|lessons|tips|signs|habits)\b"), 0.9),
    (re.compile(rf"^(stop|never|always|{DONT}|listen|imagine|look|forget|watch)\b"), 0.85),
    (re.compile(r"\b(secret|truth about|biggest mistake|myth|misconception|hack|nobody talks)\b"), 0.8),
    (re.compile(rf"\b{THATS} (the|why|how) (whole |real |only )?(secret|point|reason|answer)\b"), 0.8),
    (re.compile(r"^(why|how|what|what if|when|where|who) \w+"), 0.75),
    (re.compile(rf"\b{YOURE} (probably|doing|going to|not)\b"), 0.75),
    (re.compile(r"\b(i|we) (was|used to|never|almost|once|had to|spent)\b"), 0.6),
    (re.compile(r"^(let me|i'?ll|i want to) (tell|show|explain|walk)\b"), 0.6),
    (re.compile(rf"\b(turns out|it turns out|the reality is|actually,|{ITS} not)\b"), 0.6),
    (re.compile(r"\b(if you|when you) \w+.{0,40}\b(you'?ll|you will|you can|you should)\b"), 0.55),
    (re.compile(r"\b(this (changed|changes|is) (everything|my life|the game))\b"), 0.9),
]

# ---------------------------------------------------------------------------
# Curiosity - open loops that make the viewer stay for the answer.
# ---------------------------------------------------------------------------

CURIOSITY_MARKERS: FrozenSet[str] = frozenset(
    {
        "why", "how", "what", "whether", "wonder", "wondering", "question",
        "mystery", "surprising", "surprised", "unexpected", "weird", "strange",
        "counterintuitive", "paradox", "catch", "twist", "guess",
    }
)

CURIOSITY_PHRASES: List[Pattern[str]] = [
    re.compile(rf"\b{HERES} the (thing|catch|problem|kicker|twist)\b"),
    re.compile(rf"\bbut (wait|then|{HERES})\b"),
    re.compile(r"\bthe (thing|problem|catch) is\b"),
    re.compile(r"\bwhat (nobody|no one|most people)\b"),
    re.compile(r"\byou'?d think\b"),
    re.compile(r"\band that'?s when\b"),
]

# ---------------------------------------------------------------------------
# Payoff - does the window actually resolve what it opened?
# ---------------------------------------------------------------------------

PAYOFF_PHRASES: List[Pattern[str]] = [
    re.compile(rf"\b({THATS}|this is) (why|how|what|the reason)\b"),
    re.compile(r"\bthe (answer|reason|lesson|point|takeaway|result) is\b"),
    re.compile(r"\b(turns out|it turned out|as it turns out)\b"),
    re.compile(r"\b(so|and) (the|what) (result|happened|i learned|we found)\b"),
    re.compile(r"\bwhich (means|is why)\b"),
    re.compile(r"\b(in the end|at the end of the day|bottom line|long story short)\b"),
    re.compile(r"\b(the fix|the solution|what worked|what changed) (is|was)\b"),
    re.compile(rf"\b({HERES} what|this is what) (i|we|you) (did|do|found|learned)\b"),
    re.compile(rf"\b{THATS} the whole (secret|point|thing)\b"),
]

EMOTION_WORDS: FrozenSet[str] = frozenset(
    {
        "amazing", "incredible", "insane", "crazy", "wild", "shocking", "shocked",
        "unbelievable", "ridiculous", "absurd", "brutal", "devastating", "terrifying",
        "terrible", "horrible", "awful", "disaster", "nightmare", "furious", "angry",
        "hate", "love", "obsessed", "beautiful", "stunning", "perfect", "genius",
        "brilliant", "stupid", "dumb", "wrong", "failed", "failure", "broke", "broken",
        "died", "dying", "kill", "killed", "fight", "war", "risk", "danger", "dangerous",
        "scared", "afraid", "panic", "desperate", "shame", "proud", "cried", "crying",
        "laugh", "laughing", "hilarious", "funny", "embarrassing", "awkward",
        "expensive", "free", "million", "billion", "fortune", "bankrupt", "fired",
        "banned", "illegal", "scandal", "exposed", "caught", "secret", "lied", "lie",
    }
)

INTENSIFIERS: FrozenSet[str] = frozenset(
    {
        "literally", "actually", "absolutely", "completely", "totally", "utterly",
        "insanely", "extremely", "incredibly", "seriously", "genuinely", "massively",
        "way", "so", "such", "really", "never", "always", "every", "single", "ever",
    }
)

# ---------------------------------------------------------------------------
# Quotability - absolutes and second-person advice travel as screenshots.
# ---------------------------------------------------------------------------

QUOTABLE_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\b(never|always) \w+"),
    re.compile(r"\bthe only (thing|way|reason|question)\b"),
    re.compile(r"\b(you|people) (should|need to|have to|must|can'?t|cannot|shouldn'?t)\b"),
    re.compile(r"\bif you (want|need|do) \w+"),
    re.compile(r"\bis not (about|the)\b"),
    re.compile(rf"\b{ITS} (not|never|always) (about|that)\b"),
    re.compile(rf"\bmost people ({DONT}|never|think|assume)\b"),
]

# ---------------------------------------------------------------------------
# Structure and noise
# ---------------------------------------------------------------------------

LIST_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\b(first|firstly|number one|one)\b.{0,200}\b(second|secondly|number two|two)\b", re.S),
    re.compile(r"\b(three|four|five|two|\d+) (things|ways|reasons|steps|rules|mistakes|tips)\b"),
    re.compile(r"\bstep (one|two|three|\d+)\b"),
]

LAUGHTER_PATTERNS: List[Pattern[str]] = [
    re.compile(r"\[\s*(laughter|laughs|laughing|applause|cheering|music)\s*\]"),
    re.compile(r"\(\s*(laughter|laughs|laughing|applause)\s*\)"),
    re.compile(r"\b(haha+|hehe+|lmao|lol)\b"),
]

FILLER_WORDS: FrozenSet[str] = frozenset(
    {"um", "uh", "erm", "ah", "eh", "hmm", "mmm", "like", "basically", "literally",
     "sorta", "kinda", "anyway", "whatever", "right", "okay", "ok", "yeah", "yep"}
)

#: Openers that make a clip sound like it started in the middle of something.
DANGLING_OPENERS: FrozenSet[str] = frozenset(
    {"and", "but", "so", "because", "which", "that", "or", "then", "also",
     "however", "anyway", "although", "though", "plus", "yet", "since", "while",
     "therefore", "thus", "meanwhile", "otherwise", "besides"}
)

#: Words carrying no topical information, used for density and similarity.
STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
        "these", "those", "is", "are", "was", "were", "be", "been", "being", "am",
        "do", "does", "did", "doing", "have", "has", "had", "having", "will", "would",
        "can", "could", "should", "shall", "may", "might", "must", "i", "you", "he",
        "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
        "his", "its", "our", "their", "of", "to", "in", "on", "at", "for", "with",
        "from", "by", "about", "as", "into", "like", "through", "after", "over",
        "between", "out", "up", "down", "off", "again", "so", "just", "not", "no",
        "yes", "very", "too", "also", "here", "there", "what", "which", "who", "whom",
        "when", "where", "why", "how", "all", "any", "both", "each", "more", "most",
        "other", "some", "such", "only", "own", "same", "get", "got", "go", "going",
        "know", "think", "really", "well", "now", "one", "two", "thing", "things",
    }
)

#: Generic tags appended to every clip's suggested caption, per platform.
PLATFORM_HASHTAGS: Dict[str, List[str]] = {
    "tiktok": ["fyp", "foryou", "viral", "storytime"],
    "reels": ["reels", "instagram", "explore", "viral"],
    "shorts": ["shorts", "youtubeshorts", "viral"],
    "square": ["video", "clip"],
}

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, apostrophes preserved (``don't`` stays one word)."""
    return TOKEN_RE.findall(text.lower())


def content_words(text: str) -> List[str]:
    """Tokens that carry topic information."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]
