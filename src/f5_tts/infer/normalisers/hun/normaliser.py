import csv
import re
from num2words import num2words
import os

# Határozzuk meg a normaliser.py könyvtárát
base_dir = os.path.dirname(os.path.abspath(__file__))

def load_force_changes(filename="force_changes.csv"):
    # A fájl elérési útja a base_dir könyvtárhoz képest
    file_path = os.path.join(base_dir, filename)
    force_changes = {}
    with open(file_path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                key, value = row
                force_changes[key.strip()] = value.strip()
    return force_changes

def apply_force_changes(text, force_changes):
    # Fix cserék alkalmazása a szövegre
    for key, value in force_changes.items():
        text = text.replace(key, f' {value} ')
    return text

def load_changes(filename="changes.csv"):
    # A fájl elérési útja a base_dir könyvtárhoz képest
    file_path = os.path.join(base_dir, filename)
    changes = {}
    with open(file_path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                key, value = row
                changes[key.strip()] = value.strip()
    return changes

def apply_changes(text, changes):
    # Cserék alkalmazása csak teljes szavakra, betűmérettől függetlenül
    for key, value in changes.items():
        pattern = r'\b{}\b'.format(re.escape(key))
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)
    return text

ordinals = {
    1: 'első',
    2: 'második',
    3: 'harmadik',
    4: 'negyedik',
    5: 'ötödik',
    6: 'hatodik',
    7: 'hetedik',
    8: 'nyolcadik',
    9: 'kilencedik',
    10: 'tizedik',
    # További sorszámok hozzáadása szükség szerint
}

def replace_ordinals(text, ordinals):
    # Sorszámok átírása, kivéve ha a mondat végén vannak
    def repl(match):
        num = int(match.group(1))
        start, end = match.span()
        following_text = text[end:]
        if re.match(r'^\s*$', following_text) or re.match(r'^\s*[\.!\?]', following_text):
            return match.group(0)
        ordinal_word = ordinals.get(num, num2words(num, to='ordinal', lang='hu'))
        return ordinal_word
    pattern = r'(\d+)\.(?![\s]*$|[\s]*[\.!\?])'
    text = re.sub(pattern, repl, text)
    return text

months = {
    'jan.': 'január',
    'feb.': 'február',
    'márc.': 'március',
    'már.': 'március',
    'ápr.': 'április',
    'máj.': 'május',
    'jún.': 'június',
    'júl.': 'július',
    'aug.': 'augusztus',
    'szept.': 'szeptember',
    'szep.': 'szeptember',
    'okt.': 'október',
    'nov.': 'november',
    'dec.': 'december',
}

months_numbers = {
    1: 'január',
    2: 'február',
    3: 'március',
    4: 'április',
    5: 'május',
    6: 'június',
    7: 'július',
    8: 'augusztus',
    9: 'szeptember',
    10: 'október',
    11: 'november',
    12: 'december',
}

day_words = {
    1: 'elseje',
    2: 'másodika',
    3: 'harmadika',
    4: 'negyedike',
    5: 'ötödike',
    6: 'hatodika',
    7: 'hetedike',
    8: 'nyolcadika',
    9: 'kilencedike',
    10: 'tizedike',
    11: 'tizenegyedike',
    12: 'tizenkettedike',
    13: 'tizenharmadika',
    14: 'tizennegyedike',
    15: 'tizenötödike',
    16: 'tizenhatodika',
    17: 'tizenhetedike',
    18: 'tizennyolcadika',
    19: 'tizenkilencedike',
    20: 'huszadika',
    21: 'huszonegyedike',
    22: 'huszonkettedike',
    23: 'huszonharmadika',
    24: 'huszonnegyedike',
    25: 'huszonötödike',
    26: 'huszonhatodika',
    27: 'huszonhetedike',
    28: 'huszonnyolcadika',
    29: 'huszonkilencedike',
    30: 'harmincadika',
    31: 'harmincegyedike',
}

def day_to_text(day):
    # Napok átírása szöveges formára
    return day_words.get(day, num2words(day, lang='hu') + 'ika')

def replace_dates(text):
    # Dátumok felismerése és átírása
    month_abbrs = '|'.join(re.escape(k) for k in months.keys())

    # Év.Hónap.Nap formátum (2015.10.23.)
    pattern1 = r'(\d{4})\.(\d{1,2})\.(\d{1,2})\.'
    def repl1(match):
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        year_text = num2words(year, lang='hu')
        month_text = months_numbers.get(month, '')
        day_text = day_to_text(day)
        return f'{year_text} {month_text} {day_text}'
    text = re.sub(pattern1, repl1, text)

    # Év.HónapRöv.Nap formátum (2015.okt.23.)
    pattern2 = r'(\d{4})\.(' + month_abbrs + r')(\d{1,2})\.'
    def repl2(match):
        year = int(match.group(1))
        month_abbr = match.group(2)
        day = int(match.group(3))
        year_text = num2words(year, lang='hu')
        month_text = months.get(month_abbr.lower(), month_abbr)
        day_text = day_to_text(day)
        return f'{year_text} {month_text} {day_text}'
    text = re.sub(pattern2, repl2, text)

    # HónapRöv.Nap formátum (okt.23.)
    pattern3 = r'(' + month_abbrs + r')(\d{1,2})\.'
    def repl3(match):
        month_abbr = match.group(1)
        day = int(match.group(2))
        month_text = months.get(month_abbr.lower(), month_abbr)
        day_text = day_to_text(day)
        return f'{month_text} {day_text}'
    text = re.sub(pattern3, repl3, text)

    # HónapRöv. Nap-án formátum (okt. 23-án)
    pattern4 = r'(' + month_abbrs + r')\s+(\d{1,2})-án'
    def repl4(match):
        month_abbr = match.group(1)
        day = int(match.group(2))
        month_text = months.get(month_abbr.lower(), month_abbr)
        day_text = day_to_text(day) + 'n'
        return f'{month_text} {day_text}'
    text = re.sub(pattern4, repl4, text)

    return text

def replace_times(text):
    # Időpontok felismerése és átírása
    pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?'
    def repl(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = match.group(3)
        hour_text = num2words(hour, lang='hu')
        minute_text = num2words(minute, lang='hu')
        time_text = f'{hour_text} óra {minute_text} perc'
        if second:
            second = int(second)
            second_text = num2words(second, lang='hu')
            time_text += f' {second_text} másodperc'
        return time_text
    text = re.sub(pattern, repl, text)
    return text

def replace_numbers(text):
    # Számok átírása szöveges megfelelőjükre
    pattern = r'\b\d+\b'
    def repl(match):
        num = int(match.group(0))
        return num2words(num, lang='hu')
    text = re.sub(pattern, repl, text)
    return text

def remove_duplicate_spaces(text):
    # Többszörös szóközök eltávolítása
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_unwanted_characters(text):
    # Az eltávolítandó karakterek listája
    unwanted_characters = r'[*\-\"\'\:\(\)/#@]'
    # Eltávolítjuk az összes felsorolt karaktert
    return re.sub(unwanted_characters, ' ', text)

def add_prefix(text):
    # Hozzáadja a "... " szöveget a szöveg elejéhez
    return '... ' + text.lower()

def normalize(text):
    # A szöveg normalizálása a megadott lépésekkel
    force_changes = load_force_changes('force_changes.csv')
    changes = load_changes('changes.csv')

    text = apply_force_changes(text, force_changes)
    text = apply_changes(text, changes)
    text = replace_dates(text)
    text = replace_times(text)
    text = replace_ordinals(text, ordinals)
    text = replace_numbers(text)
    text = remove_unwanted_characters(text)
    text = remove_duplicate_spaces(text)
    text = add_prefix(text)

    return text

if __name__ == "__main__":
    # Példa szöveg
    sample_text = "Ez egy példa, KENY, szöveg 10% és 7:15 időponttal 2015.10.23. dátummal. Chartmen"
    normalized_text = normalize(sample_text)
    print(normalized_text)
