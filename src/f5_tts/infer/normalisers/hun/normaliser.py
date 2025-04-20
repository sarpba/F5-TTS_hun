import csv
import re
from num2words import num2words
import os

# Határozzuk meg a normaliser.py könyvtárát
base_dir = os.path.dirname(os.path.abspath(__file__))

# Római számokat arab számmá alakító segédfüggvény
def roman_to_int(s):
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev = 0
    for char in reversed(s):
        val = roman_map[char]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total

def replace_roman_numerals(text):
    """
    Minden olyan tokent, ami csak római számból áll (I,V,X,L,C,D,M),
    opcionálisan egy végponttal, arab számmá alakítjuk.
    Pl. 'VI' -> '6', 'IX.' -> '9.'
    """
    pattern = re.compile(r'\b([MDCLXVI]+)(\.)?\b')
    def repl(m):
        roman, dot = m.group(1), m.group(2) or ''
        # csak akkor alakítjuk, ha érvényes római számként értelmezhető
        try:
            arab = roman_to_int(roman)
        except KeyError:
            return m.group(0)
        return f"{arab}{dot}"
    return pattern.sub(repl, text)

def load_force_changes(filename="force_changes.csv"):
    """
    A force_changes.csv most négy oszlopos:
    key, value, spaces_before, spaces_after
    """
    file_path = os.path.join(base_dir, filename)
    force_changes = {}
    with open(file_path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            key, value, spaces_before, spaces_after = row
            key = key.strip()
            value = value.strip()
            # számokra castelünk
            before = int(spaces_before.strip())
            after = int(spaces_after.strip())
            force_changes[key] = (value, before, after)
    return force_changes

def apply_force_changes(text, force_changes):
    """
    A CSV-ben megadott szócsere után annyi szóközt teszünk
    előre-utólag, amennyit a 3–4. oszlopban látunk.
    """
    for key, (value, before, after) in force_changes.items():
        replacement = ' ' * before + value + ' ' * after
        text = text.replace(key, replacement)
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


def replace_ordinals(text):
    """
    Bármilyen nagyságú arab számból álló sorszámot (pl. 1233.) 
    átír num2words segítségével magyar ordítóvá.
    A patrón biztosítja, hogy a mondatvégén álló számot ponttal ne bántsa.
    """
    pattern = re.compile(r'\b(\d+)\.(?!\s*$|\s*[\.!\?])')
    def repl(m):
        num = int(m.group(1))
        # a num2words kész ordító formát ad vissza, pl. 'ezerkettőszázharmincháromadik'
        return num2words(num, to='ordinal', lang='hu')
    return pattern.sub(repl, text)

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
    # 0. Külön kezeli az "N-án" vagy "N-én" formátumot
    pattern0 = re.compile(r'\b(\d{1,2})-(án|én)\b')
    def repl0(m):
        day = int(m.group(1))
        suffix = m.group(2)       # 'án' vagy 'én'
        ordinal = num2words(day, to='ordinal', lang='hu')
        return ordinal + suffix
    text = pattern0.sub(repl0, text)

    # --- először mindegyik dátumformátum, mint eddig ---
    month_abbrs = '|'.join(re.escape(k) for k in months.keys())

    # 1. Év.Hónap.Nap formátum (2015.10.23.)
    pattern1 = re.compile(r'(\d{4})\.(\d{1,2})\.(\d{1,2})\.')
    def repl1(m):
        year, month, day = map(int, m.groups())
        year_text = num2words(year, lang='hu')
        month_text = months_numbers.get(month, '')
        day_text = day_to_text(day)
        return f'{year_text} {month_text} {day_text}'
    text = pattern1.sub(repl1, text)

    # 2. Év.HónapRöv.Nap formátum (2015.okt.23.)
    pattern2 = re.compile(r'(\d{4})\.(' + month_abbrs + r')(\d{1,2})\.')
    def repl2(m):
        year = int(m.group(1))
        month_abbr = m.group(2)
        day = int(m.group(3))
        year_text = num2words(year, lang='hu')
        month_text = months.get(month_abbr.lower(), month_abbr)
        day_text = day_to_text(day)
        return f'{year_text} {month_text} {day_text}'
    text = pattern2.sub(repl2, text)

    # 3. HónapRöv.Nap formátum (okt.23.)
    pattern3 = re.compile(r'(' + month_abbrs + r')(\d{1,2})\.')
    def repl3(m):
        month_abbr = m.group(1)
        day = int(m.group(2))
        month_text = months.get(month_abbr.lower(), month_abbr)
        day_text = day_to_text(day)
        return f'{month_text} {day_text}'
    text = pattern3.sub(repl3, text)

    # 4. HónapRöv. Nap-án formátum (okt. 23-án)
    pattern4 = re.compile(r'(' + month_abbrs + r')\s+(\d{1,2})-án')
    def repl4(m):
        month_abbr = m.group(1)
        day = int(m.group(2))
        month_text = months.get(month_abbr.lower(), month_abbr)
        day_text = day_to_text(day) + 'n'
        return f'{month_text} {day_text}'
    text = pattern4.sub(repl4, text)

    # 5. Maradék rövid hónapnevek: dec. -> december stb.
    #    (ezeket nem köti nap vagy év, csak önállóan szerepelnek)
    pattern5 = re.compile(r'(?<!\w)(' + month_abbrs + r')(?!\w)')
    def repl5(m):
        abb = m.group(1).lower()
        return months.get(abb, abb)
    text = pattern5.sub(repl5, text)

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

    text = replace_roman_numerals(text)
    text = apply_force_changes(text, force_changes)
    text = apply_changes(text, changes)
    text = replace_dates(text)
    text = replace_times(text)
    text = replace_ordinals(text)
    text = replace_numbers(text)
    text = remove_unwanted_characters(text)
    text = remove_duplicate_spaces(text)
    text = add_prefix(text)

    return text

if __name__ == "__main__":
    # Példa szöveg
    sample_text = ("Ez egy példa szöveg 2023.10.23-án, ami tartalmaz római számokat: VI, és arab számokat: 1234. "
                   "Ezen kívül van benne időpont is: 12:30:45, és dátumok is: okt. 23-án.")
    normalized_text = normalize(sample_text)
    print(normalized_text)
