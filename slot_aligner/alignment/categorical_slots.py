from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from slot_aligner.alignment.utils import find_first_in_list, get_slot_value_alternatives


def _plural(word):
    """Converts a word to its plural form.

    Credit: https://stackoverflow.com/questions/18902608/generating-the-plural-form-of-a-noun
    """
    if word.endswith('fe'):
        # wolf -> wolves
        return word[:-2] + 'ves'
    elif word.endswith('f'):
        # knife -> knives
        return word[:-1] + 'ves'
    elif word.endswith('o'):
        # potato -> potatoes
        return word + 'es'
    elif word.endswith('us'):
        # cactus -> cacti
        return word[:-2] + 'i'
    elif word.endswith('on'):
        # criterion -> criteria
        return word[:-2] + 'a'
    elif word.endswith('y'):
        # community -> communities
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'


def align_categorical_slot(text, text_tok, slot, value, mode='exact_match', allow_plural=False):
    # TODO: load alternatives only once
    alternatives = get_slot_value_alternatives(slot)

    pos = find_value_alternative(text, text_tok, value, alternatives, mode=mode, allow_plural=allow_plural)

    return pos


def find_value_alternative(text, text_tok, value, alternatives, mode, allow_plural=False):
    leftmost_pos = -1

    # Parse the item into tokens according to the selected mode
    if mode == 'first_word':
        value_alternatives = [value.split(' ')[0]]  # Single-element list
    elif mode == 'any_word':
        value_alternatives = value.split(' ')  # List of elements
    elif mode == 'all_words':
        value_alternatives = [value.split(' ')]  # List of single-element lists
        # print(value)
        # print(value_alternatives)
    else:
        value_alternatives = [value]  # Single-element list

    # Merge the tokens with the item's alternatives
    # print(value_alternatives)
    # print(value)
    # print(1)
    # print(value_alternatives)
    # print(2)
    if value in alternatives:
        value_alternatives += alternatives[value]
        print(value_alternatives)
        print(1)

    if allow_plural:
        value_alternatives += [_plural(value_alt) for value_alt in value_alternatives]

    # Iterate over individual tokens of the item
    for value_alt in value_alternatives:
        print(value_alt)
        # If the item is composed of a single token, convert it to a single-element list
        if not isinstance(value_alt, list):
            value_alt = [value_alt]


        # Keep track of the positions of all the item's tokens
        positions = []
        for tok in value_alt:
            if len(tok) > 4 or ' ' in tok:
                # Search for long and multi-word values in the string representation
                pos = text.find(tok)
            else:
                # Search for short single-word values in the tokenized representation
                _, pos = find_first_in_list(tok, text_tok)
            positions.append(pos)

        # If all tokens of one of the value's alternatives are matched, record the match and break
        if all([p >= 0 for p in positions]):
            leftmost_pos = min(positions)
            break
        # print(leftmost_pos)

    return leftmost_pos


# TODO @food has 24 failures which are acceptable to remove the slot
def foodSlot(text, text_tok, value):
    value = value.lower()

    pos = text.find(value)
    if pos >= 0:
        return pos
    elif value == 'english':
        return text.find('british')
    elif value == 'fast food':
        return text.find('american style')
    else:
        text_tok = word_tokenize(text)
        for token in text_tok:
            # FIXME warning this will be slow on start up
            synsets = wordnet.synsets(token, pos='n')
            synset_ctr = 0

            for synset in synsets:
                synset_ctr += 1
                hypernyms = synset.hypernyms()

                # If none of the first 3 meanings of the word has "food" as hypernym, then we do not want to
                #   identify the word as food-related (e.g. "center" has its 14th meaning associated with "food",
                #   or "green" has its 7th meaning accociated with "food").
                while synset_ctr <= 3 and len(hypernyms) > 0:
                    lemmas = [l.name() for l in hypernyms[0].lemmas()]

                    if 'food' in lemmas:
                        # DEBUG PRINT
                        # print(token)

                        return text.find(token)
                    # Skip false positives (e.g. "a" in the meaning of "vitamin A" has "food" as a hypernym,
                    #   or "coffee" in "coffee shop" has "food" as a hypernym). There are still false positives
                    #   triggered by proper nouns containing a food term, such as "Burger King" or "The Golden Curry".
                    elif 'vitamin' in lemmas:
                        break
                    elif 'beverage' in lemmas:
                        break

                    # Follow the hypernyms recursively up to the root
                    hypernyms = hypernyms[0].hypernyms()

    return pos
