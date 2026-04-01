"""
Replace all nouns in GSM8K problems with generated nonsense words.

Purpose: detect contamination and semantic shortcuts in LLM math solvers.
If an LLM's score drops on nonsense-noun versions, the delta is the
contamination + semantic shortcut signal. A deterministic solver's
score should be identical on both versions.

Approach: curated noun classes extracted from GSM8K corpus, replaced with
phonotactically valid nonsense words. No NLP dependencies.

Usage:
    python benchmark/scripts/nonsense_nouns.py benchmark/suites/gsm8k_500.json
    # writes benchmark/suites/gsm8k_500_nonsense.json
"""

import json
import re
import sys
import random
from pathlib import Path

random.seed(42)  # reproducible nonsense

# --- Nonsense word generator ---

ONSETS = [
    "br", "cr", "dr", "fr", "gr", "kr", "pr", "tr", "bl", "cl", "fl",
    "gl", "pl", "sk", "sl", "sn", "sp", "st", "sw", "qu", "z", "v",
    "th", "sh", "ch", "wh", "ph",
]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ou", "ea", "oo", "au"]
SUFFIXES_PERSON = ["a", "o", "i", "u", "ee", "ak", "el", "is", "on", "uk"]
SUFFIXES_THING = ["le", "ik", "op", "az", "ul", "ek", "om", "ux", "id", "ot"]
SUFFIXES_PLACE = ["burg", "ton", "vale", "stead", "wick", "ford", "dale", "moor"]

_used = set()


def _gen_word(suffixes, min_syl=1, max_syl=2):
    for _ in range(500):
        n = random.randint(min_syl, max_syl)
        parts = []
        for _ in range(n):
            parts.append(random.choice(ONSETS))
            parts.append(random.choice(VOWELS))
        parts.append(random.choice(suffixes))
        word = "".join(parts)
        if word.lower() not in _used:
            _used.add(word.lower())
            return word
    raise RuntimeError("exhausted nonsense word space")


def gen_person():
    return _gen_word(SUFFIXES_PERSON)


def gen_thing():
    return _gen_word(SUFFIXES_THING, 1, 2)


def gen_place():
    return _gen_word(SUFFIXES_PLACE, 1, 2)


# --- Curated noun classes from GSM8K corpus ---
# These are domain nouns that carry semantic content an LLM could exploit.
# Structural/math words (day, hour, price, total, etc.) are kept intact.

PEOPLE = {
    "janet", "john", "james", "billy", "peter", "mike", "mary", "mark",
    "tom", "tim", "bob", "bill", "sam", "dan", "joe", "jack", "ben",
    "josh", "luke", "alex", "anna", "amy", "emma", "lily", "lisa",
    "sara", "sarah", "jane", "kate", "kim", "sue", "alan", "carl",
    "dave", "dean", "eric", "fred", "gary", "greg", "hans", "ivan",
    "jeff", "ken", "kurt", "lee", "matt", "neil", "nick", "oscar",
    "paul", "phil", "ralph", "rick", "rob", "roger", "ron", "steve",
    "ted", "terry", "tony", "vic", "wade", "will", "toula", "mishka",
    "uriah", "vincent", "carlos", "toulouse", "charleston", "seattle",
    "eliza", "natalia", "emmy", "liza", "claire", "martha", "tina",
    "weng", "betty", "julie", "mabel", "carol", "gloria", "greta",
    "heather", "holly", "irene", "jean", "joyce", "karen", "kathy",
    "laura", "marie", "nancy", "norma", "pam", "rachel", "rosa",
    "ruth", "tara", "wendy", "mr", "mrs", "dr",
}

PLACES = {
    "store", "school", "farm", "park", "bakery", "market", "shop",
    "library", "hospital", "restaurant", "gym", "beach", "zoo",
    "garden", "factory", "office", "warehouse", "garage", "kitchen",
    "classroom", "stadium", "theater", "cinema", "museum", "church",
    "station", "airport", "harbor", "lake", "river", "mountain",
    "city", "town", "village", "country", "island", "forest",
    "aquarium", "cafeteria", "carnival", "circus", "clinic",
    "playground", "ranch", "shelter", "studio", "temple",
}

ANIMALS = {
    "duck", "ducks", "sheep", "chicken", "chickens", "dog", "dogs",
    "cat", "cats", "fish", "bird", "birds", "horse", "horses",
    "cow", "cows", "pig", "pigs", "rabbit", "rabbits", "goat", "goats",
    "lamb", "lambs", "turkey", "turkeys", "frog", "frogs",
    "turtle", "turtles", "kitten", "kittens", "puppy", "puppies",
    "parrot", "parrots", "hamster", "hamsters", "mouse", "mice",
    "snake", "snakes", "lizard", "lizards", "bee", "bees",
    "butterfly", "butterflies", "ant", "ants", "spider", "spiders",
}

FOOD = {
    "apple", "apples", "orange", "oranges", "cookie", "cookies",
    "cupcake", "cupcakes", "donut", "donuts", "cake", "cakes",
    "pie", "pies", "bread", "loaf", "loaves", "sandwich", "sandwiches",
    "pizza", "muffin", "muffins", "candy", "candies", "chocolate",
    "lollipop", "lollipops", "lemonade", "soda", "juice", "milk",
    "egg", "eggs", "potato", "potatoes", "carrot", "carrots",
    "tomato", "tomatoes", "banana", "bananas", "grape", "grapes",
    "strawberry", "strawberries", "cherry", "cherries", "peach", "peaches",
    "watermelon", "melon", "corn", "rice", "pasta", "noodle", "noodles",
    "cereal", "pancake", "pancakes", "waffle", "waffles", "taco", "tacos",
    "burger", "burgers", "fries", "steak", "bacon", "ham", "cheese",
    "pastry", "pastries", "brownie", "brownies", "cracker", "crackers",
    "pretzel", "pretzels", "popcorn", "peanut", "peanuts",
    "fruit", "fruits", "vegetable", "vegetables",
}

OBJECTS = {
    "book", "books", "card", "cards", "marble", "marbles",
    "toy", "toys", "pencil", "pencils", "pen", "pens",
    "ball", "balls", "bottle", "bottles", "candle", "candles",
    "flower", "flowers", "shirt", "shirts", "shoe", "shoes",
    "hat", "hats", "jean", "jeans", "dress", "dresses",
    "sock", "socks", "glove", "gloves", "scarf", "scarves",
    "ring", "rings", "necklace", "bracelet", "watch", "watches",
    "phone", "computer", "laptop", "tablet", "camera",
    "bike", "bicycle", "car", "cars", "truck", "trucks",
    "bus", "van", "boat", "ship", "plane", "train", "trains",
    "ticket", "tickets", "stamp", "stamps", "coin", "coins",
    "key", "keys", "lamp", "lamps", "chair", "chairs",
    "table", "tables", "bed", "beds", "desk", "desks",
    "cup", "cups", "plate", "plates", "bowl", "bowls",
    "spoon", "fork", "knife", "basket", "baskets",
    "bucket", "buckets", "rope", "ropes", "ladder", "ladders",
    "tool", "tools", "nail", "nails", "brick", "bricks",
    "stone", "stones", "rock", "rocks", "stick", "sticks",
    "leaf", "leaves", "seed", "seeds", "plant", "plants",
    "tree", "trees", "bush", "bushes", "vine", "vines",
    "page", "pages", "letter", "letters", "envelope", "envelopes",
    "painting", "paintings", "photo", "photos", "picture", "pictures",
    "sticker", "stickers", "balloon", "balloons", "ribbon", "ribbons",
    "animal", "animals", "doll", "dolls", "puzzle", "puzzles",
    "game", "games", "block", "blocks", "lego", "crayon", "crayons",
    "bar", "bars", "can", "cans", "jar", "jars", "pack", "packs",
    "container", "containers", "tray", "trays", "shelf", "shelves",
    "towel", "towels", "blanket", "blankets", "pillow", "pillows",
    "bead", "beads", "button", "buttons", "thread", "string",
    "wire", "cable", "battery", "batteries",
    "robe", "fiber",
}

ROLES = {
    "teacher", "student", "students", "doctor", "nurse", "farmer",
    "baker", "chef", "driver", "pilot", "captain", "soldier",
    "player", "players", "coach", "dancer", "singer", "artist",
    "painter", "writer", "reader", "worker", "workers",
    "boy", "boys", "girl", "girls", "child", "children", "kid", "kids",
    "baby", "babies", "man", "men", "woman", "women", "people", "person",
    "brother", "sister", "mother", "father", "mom", "dad",
    "son", "daughter", "uncle", "aunt", "cousin", "grandma", "grandpa",
    "grandmother", "grandfather", "parent", "parents",
    "friend", "friends", "neighbor", "neighbors", "guest", "guests",
    "customer", "customers", "client", "clients",
    "team", "class", "family", "crew", "staff",
}

MISC_NOUNS = {
    "water", "rain", "snow", "ice", "sand", "dirt", "mud", "grass",
    "field", "road", "street", "highway", "bridge", "tunnel",
    "building", "house", "room", "floor", "wall", "door", "window",
    "roof", "pool", "pond", "fence", "gate",
    "trip", "journey", "ride", "race",
    "party", "birthday", "wedding", "vacation", "holiday", "festival",
    "homework", "exam", "quiz",
    "recipe", "meal", "dinner", "lunch", "breakfast", "snack",
    "delivery", "shipment", "supply",
    "salary", "rent", "tax", "tip", "fee",
    "profit", "debt", "loan", "deposit",
    "company", "business",
    "color", "shape",
    "hair", "eye", "face",
    "distance",
}

# Words that appear in noun lists but are commonly used as verbs/adjectives
# in word problems — must be excluded to avoid breaking sentence structure
AMBIGUOUS_SKIP = {
    "will", "can", "plant", "plants", "order", "run", "walk",
    "match", "fine", "bill", "well", "game", "step", "steps",
    "point", "points", "score", "level", "grade", "rank",
    "lap", "laps", "tank", "fuel", "gas", "oil",
    "hand", "arm", "leg", "head", "name", "age", "size",
    "loss", "load", "test", "task", "project", "job",
    "bank", "deposit", "payment", "fee",
}

# Build category-to-generator mapping
# Each source word maps to (category, singular_form)
_noun_registry = {}

def _register(words, category):
    for w in words:
        _noun_registry[w.lower()] = category

_register(PEOPLE, "person")
_register(PLACES, "place")
_register(ANIMALS, "thing")
_register(FOOD, "thing")
_register(OBJECTS, "thing")
_register(ROLES, "person")
_register(MISC_NOUNS, "thing")


def _get_replacement(word, mapping):
    """Get or create a nonsense replacement for a word."""
    key = word.lower()
    if key in AMBIGUOUS_SKIP:
        return None
    if key in mapping:
        return mapping[key]

    # Check if it's a plural and the singular is mapped
    if key.endswith("es") and key[:-2] in mapping:
        base = mapping[key[:-2]]
        mapping[key] = base + "es"
        return mapping[key]
    if key.endswith("s") and key[:-1] in mapping:
        base = mapping[key[:-1]]
        mapping[key] = base + "s"
        return mapping[key]

    category = _noun_registry.get(key)
    if not category:
        return None

    # Generate based on category
    if category == "person":
        replacement = gen_person()
    elif category == "place":
        replacement = gen_place()
    else:
        replacement = gen_thing()

    mapping[key] = replacement

    # Pre-generate plural forms
    if not key.endswith("s"):
        if key + "s" in _noun_registry:
            mapping[key + "s"] = replacement.lower() + "s"
        if key + "es" in _noun_registry:
            mapping[key + "es"] = replacement.lower() + "es"

    return replacement


def replace_nouns(text, mapping):
    """Replace all known nouns in text with nonsense words."""

    # First handle multi-word proper nouns (e.g., "San Rafael")
    # Split into sentences, then process
    def _replace_word(match):
        word = match.group(0)
        key = word.lower()
        replacement = _get_replacement(word, mapping)
        if replacement is None:
            return word
        # Preserve case
        if word[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement.lower()

    # Match whole words only
    result = re.sub(r'\b[A-Za-z]+\b', _replace_word, text)
    return result


def transform_suite(input_path):
    """Transform an entire benchmark suite."""
    input_path = Path(input_path)
    output_path = input_path.with_name(
        input_path.stem + "_nonsense" + input_path.suffix
    )

    with open(input_path) as f:
        suite = json.load(f)

    new_cases = []
    stats = {"total": 0, "replaced": 0, "nouns_found": 0}

    for case in suite["cases"]:
        # Fresh mapping per problem so nouns are unique across problems
        mapping = {}
        new_q = replace_nouns(case["question"], mapping)

        new_case = {
            **case,
            "question": new_q,
            "original_question": case["question"],
            "noun_mapping": {k: v for k, v in mapping.items()
                            if k != v.lower()},  # only actual replacements
        }
        new_cases.append(new_case)
        stats["total"] += 1
        if mapping:
            stats["replaced"] += 1
            stats["nouns_found"] += len(mapping)

    suite["cases"] = new_cases
    suite["name"] = suite["name"] + "_nonsense"
    suite["description"] = (
        suite["description"]
        + " — nouns replaced with nonsense words to detect contamination"
    )

    with open(output_path, "w") as f:
        json.dump(suite, f, indent=2)

    print(f"Wrote {len(new_cases)} problems to {output_path}")
    print(f"Problems with replacements: {stats['replaced']}/{stats['total']}")
    print(f"Total noun mappings: {stats['nouns_found']}")
    print(f"\n--- Sample transformations ---\n")
    for case in new_cases[:10]:
        if case["noun_mapping"]:
            print(f"ORIGINAL: {case['original_question'][:200]}")
            print(f"NONSENSE: {case['question'][:200]}")
            print(f"MAPPING:  {case['noun_mapping']}")
            print()

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nonsense_nouns.py <suite.json>")
        sys.exit(1)
    transform_suite(sys.argv[1])
