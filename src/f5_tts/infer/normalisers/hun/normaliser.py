import csv
import re
from num2words import num2words
import os
import random

# Határozzuk meg a normaliser.py könyvtárát
base_dir = os.path.dirname(os.path.abspath(__file__))

# Betűk kiejtése mozaikszavakhoz és alfanumerikus szavakhoz
letter_pronunciations = {
    'A': 'a', 'B': 'bé', 'C': 'cé', 'D': 'dé', 'E': 'e', 'F': 'ef', 'G': 'gé', 'H': 'há',
    'I': 'í', 'J': 'jé', 'K': 'ká', 'L': 'el', 'M': 'em', 'N': 'en', 'O': 'ó', 'P': 'pé',
    'Q': 'kú', 'R': 'er', 'S': 'ess', 'T': 'té', 'U': 'ú', 'V': 'vé', 'W': 'dupla vé',
    'X': 'iksz', 'Y': 'ipszilon', 'Z': 'zé',
    'Á': 'á', 'É': 'é', 'Í': 'í', 'Ó': 'ó', 'Ö': 'ö', 'Ő': 'ő', 'Ú': 'ú', 'Ü': 'ü', 'Ű': 'ű'
}

def pronounce_letters(word):
    return ' '.join(letter_pronunciations.get(char.upper(), char) for char in word)

def replace_acronyms(text):
    """
    Mozaikszavak (csak nagybetűkből álló szavak) betűzése.
    """
    # Kizárjuk a római számokat, hogy ne legyenek betűzve
    roman_numerals_pattern = r'[IVXLCDM]+'
    # A pattern biztosítja, hogy csak olyan szavakat célozzon, amelyek legalább egy nem római szám karaktert is tartalmaznak,
    # vagy ha csak római karakterekből állnak, akkor ne legyenek tisztán római számok (pl. MIX, de nem MCM).
    # Ez a feltétel bonyolult lehet, egyszerűbb a római számok explicit kizárása.
    pattern = re.compile(r'\b(?!(?:' + roman_numerals_pattern + r')\b)([A-ZÁÉÍÓÖŐÚÜŰ]{2,})\b') # Legalább két nagybetűs mozaikszavak
    def repl(m):
        acronym = m.group(1)
        # Ellenőrizzük, hogy a mozaikszó nem egyezik-e meg egy ismert római számmal,
        # amit a replace_roman_numerals nem alakított át (mert pl. nincs utána pont).
        # Ez a lépés elhagyható, ha a replace_roman_numerals minden római számot kezelne,
        # vagy ha elfogadjuk, hogy a pont nélküli római számok (pl. "XIV") betűzve lesznek, ha nagybetűsek.
        # A jelenlegi feladatleírás szerint csak a ponttal végződő római számokat kell átírni.
        # Tehát egy "XIV" (pont nélkül) itt mozaikszónak minősülhet, ha a kizárás nem elég erős.
        # A (?!(?:[IVXLCDM]+)\b) kizárásnak ezt kezelnie kellene.
        return pronounce_letters(acronym)
    return pattern.sub(repl, text)

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
    Minden olyan római számot átír, ami után pont van.
    Pl. 'IX.' -> '9.', 'IV.' -> '4.'
    """
    # Minta: minden római számot keres, ami után pont van
    pattern = re.compile(r'([IVXLCDM]+)\.')
    def repl(m):
        roman = m.group(1)
        try:
            arab = roman_to_int(roman)
            return f"{arab}."
        except KeyError: # Ha nem érvényes római szám, hagyjuk változatlanul
            return m.group(0)
    return pattern.sub(repl, text)

def replace_alphanumeric(text):
    """
    Olyan szavak szétbontása és átírása, amik betűket és számokat egyaránt tartalmaznak.
    A betűket betűzve, a számokat szövegesen írja le.
    Pl. 'A123B' -> 'á százhuszonhárom bé', 'K-9' -> 'ká kilenc'
    """
    # Ez a minta olyan szavakat keres, amelyek tartalmaznak legalább egy betűt és legalább egy számot,
    # és tartalmazhatnak kötőjelet is.
    pattern = re.compile(r'\b([A-Za-zÁÉÍÓÖŐÚÜŰ\d-]*([A-Za-zÁÉÍÓÖŐÚÜŰ]+[A-Za-zÁÉÍÓÖŐÚÜŰ\d-]*\d+|\d+[A-Za-zÁÉÍÓÖŐÚÜŰ\d-]*[A-Za-zÁÉÍÓÖŐÚÜŰ]+)[A-Za-zÁÉÍÓÖŐÚÜŰ\d-]*)\b')

    def repl(m):
        word = m.group(0)
        # Felbontjuk a szót egyes karakterekre
        parts = []
        current = ''
        # Új tokenizáló logika: kötőjel utáni szavakat ne bontsuk fel
        tokens = re.split(r'(-)', word)  # Kötőjelek mentén vág, de megtartja a kötőjeleket
        for token in tokens:
            if not token:
                continue
            if token == '-':
                parts.append(token)
                continue
                
            # Felbontás betű-szám határoknál
            sub_tokens = re.split(r'(\d+)', token)  # Számok mentén vág
            for sub_token in sub_tokens:
                if not sub_token:
                    continue
                parts.append(sub_token)
        if current:
            parts.append(current)
        
        result_parts = []
        for idx, part in enumerate(parts):
            if part == '-':
                # Kötőjel esetén szóközt adunk hozzá
                result_parts.append(' ')
            elif part.isalpha():
                if idx > 0 and parts[idx-1] == '-':
                    # Csak kisbetűs suffix-eket hagyjuk változatlanul
                    if part.islower() and part in ['es', 'os', 'as', 'ert', 'ig', 'bol', 'rol']:
                        result_parts.append(part)
                    else:
                        # Nagybetűs suffix-eket betűzzük
                        result_parts.append(pronounce_letters(part))
                else:
                    result_parts.append(pronounce_letters(part))
            elif part.isdigit():
                result_parts.append(num2words(int(part), lang='hu'))
            else:
                # Egyéb karakterek megtartása
                result_parts.append(part)
        # Az összes elem szóközzel elválasztva
        return ' '.join(result_parts)
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
    A CSV-ben megadott szócsere minden előfordulására alkalmazza a változtatást,
    akár szó közepén is.
    """
    for key, (value, before, after) in force_changes.items():
        replacement = ' ' * before + value + ' ' * after
        # Regex használata az összes előfordulás cseréjéhez
        text = re.sub(re.escape(key), replacement, text)
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
    Kizárja az éveket, hogy ne alakítsa át sorszámmá.
    """
    # Kizárjuk az éveket (4 számjegyű számok ponttal)
    pattern = re.compile(r'\b(\d{1,3}|\d{5,})\.(?!\s*$|\s*[\.!\?])')
    def repl(m):
        num = int(m.group(1))
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
    1: 'január', 'I': 'január',
    2: 'február', 'II': 'február',
    3: 'március', 'III': 'március',
    4: 'április', 'IV': 'április',
    5: 'május', 'V': 'május',
    6: 'június', 'VI': 'június',
    7: 'július', 'VII': 'július',
    8: 'augusztus', 'VIII': 'augusztus',
    9: 'szeptember', 'IX': 'szeptember',
    10: 'október', 'X': 'október',
    11: 'november', 'XI': 'november',
    12: 'december', 'XII': 'december',
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

def day_to_text(day_num):
    # Napok átírása szöveges formára, pl. 1 -> elseje
    return day_words.get(day_num, num2words(day_num, lang='hu') + 'ika') # Fallback, bár 1-31 között nem kellene

def format_date_text_new(year_num, month_name_str, day_num):
    """
    Formázza a dátumot szövegesen: "év hónap napadik".
    year_num: int (pl. 2025)
    month_name_str: string (teljes hónapnév, pl. "június")
    day_num: int (pl. 1)
    """
    # Az év kardinális számként (nem sorszám)
    year_text = num2words(year_num, lang='hu')
    # A nap sorszámként (tizedike, elseje stb.)
    day_text = day_to_text(day_num)
    return f'{year_text} {month_name_str} {day_text}'

def replace_dates(text):
    # Segédfüggvény a dátumok szöveges formára alakításához (az új, egységesített formátum_date_text_new-t használva)

    # Kombinált hónap regexek
    # Fontos, hogy a leghosszabb hónapnevek legyenek elöl a regexben, hogy elkerüljük a részleges egyezéseket
    # pl. "júl." vs "július". A re.escape szükséges.
    sorted_month_values = sorted(months.values(), key=len, reverse=True)
    month_names_hu_regex = '|'.join(re.escape(m_val) for m_val in sorted_month_values)
    
    sorted_month_keys = sorted(months.keys(), key=len, reverse=True)
    month_abbrs_hu_regex = '|'.join(re.escape(k) for k in sorted_month_keys)
    
    roman_numerals_regex = r'(M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))' # Római számok regex, capturing group hozzáadva

    # Dátumformátumok kezelése
    patterns = [
        # 1. 2025. június 1. (teljes hónapnévvel)
        (r'(\d{4})\.\s*(' + month_names_hu_regex + r')\s*(\d{1,2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), m.group(2), int(m.group(3)))),
        # 2. 2025. 06. 01. (számmal írt hónap)
        (r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[int(m.group(2))], int(m.group(3)))),
        # 3. 2025.06.01. (számmal írt hónap, szóköz nélkül)
        (r'(\d{4})\.(\d{2})\.(\d{2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[int(m.group(2))], int(m.group(3)))),
        # 4. 2025. jún. 1. (rövidített hónapnévvel)
        (r'(\d{4})\.\s*(' + month_abbrs_hu_regex + r')\s*(\d{1,2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months[m.group(2).lower()], int(m.group(3)))),
        # 5. 2025.jún.1. (rövidített hónapnévvel, szóköz nélkül)
        (r'(\d{4})\.(' + month_abbrs_hu_regex + r')(\d{1,2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months[m.group(2).lower()], int(m.group(3)))),
        # 6. 2025. VI. 1. (római számos hónap)
        (r'(\d{4})\.\s*(' + roman_numerals_regex + r')\s*(\d{1,2})\.(?!\d)',
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[roman_to_int(m.group(2).upper())], int(m.group(3)))),
        # 7. 2025.VI.1. (római számos hónap, szóköz nélkül)
        (r'(\d{4})\.(' + roman_numerals_regex + r')(\d{1,2})\.(?!\d)',
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[roman_to_int(m.group(2).upper())], int(m.group(3)))),
        # 8. 2025-06-01 (kötőjeles formátum)
        (r'(\d{4})-(\d{2})-(\d{2})(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[int(m.group(2))], int(m.group(3)))),
        # 9. 2024. aug. 10-én (rövidített hónapnévvel, kötőjeles toldalékkal)
        (r'(\d{4})\.\s*(' + month_abbrs_hu_regex + r')\s*(\d{1,2})-([a-záéíóöőúüű]+)\b', 
         lambda m: format_date_text_new(int(m.group(1)), months[m.group(2).lower()], int(m.group(3))) + m.group(4))
    ]

    for pattern_str, repl_func in patterns:
        text = re.sub(pattern_str, repl_func, text)

    # Külön kezeli az "N-án" vagy "N-én" formátumot (ez már létezett)
    pattern_day_suffix = re.compile(r'\b(\d{1,2})-(án|én)\b')
    def repl_day_suffix(m):
        day = int(m.group(1))
        suffix = m.group(2)
        ordinal = num2words(day, to='ordinal', lang='hu')
        return ordinal + suffix
    text = pattern_day_suffix.sub(repl_day_suffix, text)
    
    # Maradék rövid hónapnevek: dec. -> december stb.
    # (ezeket nem köti nap vagy év, csak önállóan szerepelnek)
    month_abbrs_only = '|'.join(re.escape(k) for k in months.keys())
    pattern_month_only = re.compile(r'(?<!\w)(' + month_abbrs_only + r')(?!\w)')
    def repl_month_only(m):
        abb = m.group(1).lower()
        return months.get(abb, abb)
    text = pattern_month_only.sub(repl_month_only, text)

    # Maradék rövid hónapnevek: dec. -> december stb.
    # (ezeket nem köti nap vagy év, csak önállóan szerepelnek)
    month_abbrs_only = '|'.join(re.escape(k) for k in months.keys())
    pattern_month_only = re.compile(r'(?<!\w)(' + month_abbrs_only + r')(?!\w)')
    def repl_month_only(m):
        abb = m.group(1).lower()
        return months.get(abb, abb)
    text = pattern_month_only.sub(repl_month_only, text)

    return text

def replace_times(text):
    """
    Időpontok átírása két formátum közül véletlenszerűen választva:
    1. "óra perc" forma (pl. "hét óra harminc perc")
    2. "óra perc" forma (pl. "hét harminc")
    A másodperceket külön hozzáadjuk (pl. "15 óra negyvenöt perc harminc másodperc")
    Ha van másodperc, akkor mindig az első formátumot használjuk.
    """
    pattern = re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?(-kor)?\b')
    def repl(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = match.group(3)
        has_kor = match.group(4) == '-kor'
        
        hour_text = num2words(hour, lang='hu')
        minute_text = num2words(minute, lang='hu') if minute != 0 else ""

        # Ha van másodperc, akkor az első formátumot használjuk (óra, perc, másodperc)
        if second:
            second_val = int(second)
            second_text = num2words(second_val, lang='hu')
            time_str = f'{hour_text} óra {minute_text} perc {second_text} másodperc'
        else:
            # Ha a perc 00, akkor csak az órát írjuk ki
            if minute == 0:
                time_str = f'{hour_text} óra'
            else:
                # Véletlenszerű választás a két formátum között
                if random.choice([True, False]):
                    time_str = f'{hour_text} óra {minute_text} perc'
                else:
                    time_str = f'{hour_text} {minute_text}'
        
        # "kor" hozzáadása, ha szükséges
        if has_kor:
            time_str += 'kor'  # Szóköz nélkül csatoljuk
        
        return time_str
    text = pattern.sub(repl, text)
    return text

def replace_numbers(text):
    # Számok átírása szöveges megfelelőjükre, figyelve a mondatvégi számokra
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
    # Az eltávolítandó karakterek listája (kötőjel kivéve)
    unwanted_characters = r'[*\"\'\:\(\)\/#@\[\]\{\}]'
    # Eltávolítjuk az összes felsorolt karaktert
    return re.sub(unwanted_characters, ' ', text)

def add_prefix(text):
    # Hozzáadja a "... " szöveget a szöveg elejéhez
    return '... ' + text.lstrip()

def convert_to_lowercase(text):
    """Az egész szöveg kisbetűssé alakítása"""
    return text.lower()

def normalize(text):
    # A szöveg normalizálása a megadott lépésekkel
    force_changes = load_force_changes('force_changes.csv')
    changes = load_changes('changes.csv')

    text = replace_roman_numerals(text) # Római számok arabra (pl. IV. -> 4.)
    # Időpontok kezelése előbb, hogy a "-kor" még változatlan legyen
    text = replace_times(text)
    text = apply_force_changes(text, force_changes)
    text = apply_changes(text, changes)
    text = replace_acronyms(text)
    text = replace_alphanumeric(text)
    text = replace_dates(text) # Dátumok átalakítása
    text = replace_ordinals(text) # Sorszámok (pl. 4. -> negyedik)
    text = replace_numbers(text) # Számok szöveggé
    
    text = remove_unwanted_characters(text) 
    # Kivételek kezelése kötőjelekkel
    exceptions = {
        "egy-egy": "egy egy",
        "két-két": "két két",
        "három-három": "három három",
        "négy-négy": "négy négy",
        "öt-öt": "öt öt"
    }
    
    for pattern, replacement in exceptions.items():
        text = re.sub(r'\b' + re.escape(pattern) + r'\b', replacement, text)
    
    # Kötőjelek eltávolítása (szóköz helyett egybeírás)
    text = re.sub(r'-', '', text)
    
    text = remove_duplicate_spaces(text)
    text = add_prefix(text)
    text = convert_to_lowercase(text)

    return text

if __name__ == "__main__":
    # Példa szöveg az összes funkció tesztelésére
    test_cases = {
        "Római számok": "A IV. és IX. fejezet fontos. MCMXCVI. év.",
        "Force changes": "Ez ninjutsu és chips.",
        "Changes (különálló szavak)": "Az AI és a GPU fontos, like that.",
        "Mozaikszavak": "A NATO és az ENSZ ülésezik. Az EU is. USA.",
        "Alfanumerikus szavak": "Ez egy B2 vitamin és C3PO robot. Az X-Wing és a T-1000 is itt van. K9-es egység.",
        "Dátumok": "Találkozó 2025. június 1. napján. Másik dátum: 2025. 07. 15. és 2024.12.24. Továbbiak: 2023. márc. 8., 2022.okt.10., 1999. XII. 31. és 2000.I.1. Végül: 2021-05-20.",
        "Időpontok": "Reggel 7:30-kor kelek, de 08:00-kor indulok. Délután 14:05 van, este 22:15:30-kor fekszem.",
        "Római számok pont nélkül": "Ez egy IV. fejezet, de XIV Lajos nem pont nélkül.",
        "Force changes szó közepén": "Ez egy ninjutsu technika és chips-ek.",
        "Sorszámok": "Az 1. helyezett, a 23. versenyző és a 100. évforduló.",
        "Számok (kardinális)": "Vettem 1234 almát és 567 körtét. Van 0 darab.",
        "Fölösleges karakterek": "Ez *egy* \"szöveg\" (sok) fölösleges: karakterrel/. #@jel",
        "Dupla szóközök": "Itt   van  néhány  extra   szóköz.",
        "Összetett mondat": "A KFT. 2024. aug. 10-én 10:30-kor tartja a X. közgyűlését a B42-es teremben, ahol az R2D2 projekt eredményeit ismertetik.",
        "Mondatvégi pont sorszámnál": "Ez a 12. oldal. A következő a 13. oldal.",
        "Szöveg eleje": "   Ez egy szöveg szóközökkel az elején.",
        "Római számok önmagukban (nem alakulnak át)": "XIV Lajos, IV Béla.",
        "Alfanumerikus kötőjellel": "Ez egy teszt A-1, B2-C, D-3E."
    }

    # A normalizálási lépések sorrendje a normalize() függvényben definiált
    # Itt csak az egyes funkciókat teszteljük izoláltabban is, ha szükséges,
    # de a fő teszt a normalize() hívása.

    print("--- Teljes Normalizálási Teszt ---")
    full_sample_text = (
        "Ez egy példa szöveg a NATO-tól. Római számok: IV., IX., XII. "
        "Force changes: ninjutsu, chips. Changes: AI, GPU, like. "
        "Alfanumerikus: v3, r1, A123B, K-9. "
        "Dátumok: 2025. június 1., 2025. 06. 01., 2025.06.01., 2025. jún. 1., 2025.jún.1., 2025. VI. 1., 2025.VI.1., 2025-06-01. "
        "Időpont: 7:30-kor, 14:05, 09:00. "
        "Sorszámok: 1., 23., 100. Ez a 42. pont. "
        "Számok: 1234, 567, 0. "
        "Fölösleges karakterek: * - \" ' : ( ) / # @. "
        "Dupla  szóközök  itt  vannak. "
        "Vége."
    )
    normalized_full_text = normalize(full_sample_text)
    print(f"Eredeti: {full_sample_text}")
    print(f"Normalizált: {normalized_full_text}\n")

    print("--- Egyedi Tesztesetek ---")
    for desc, text_to_normalize in test_cases.items():
        print(f"Teszt: {desc}")
        print(f"  Eredeti: {text_to_normalize}")
        normalized_text = normalize(text_to_normalize)
        print(f"  Normalizált: {normalized_text}\n")

    # Speciális teszt a replace_alphanumeric-hez
    print("--- Alfanumerikus Speciális Teszt ---")
    alphanum_test_text = "A1, B2B, C3C3, D4D4D, E5-F6, G7H8I9, J10K"
    
    # Új teszt a szöveg elejével
    print("--- Szöveg elejének ellenőrzése ---")
    prefix_test = "   Ez egy szöveg szóközökkel."
    print(f"  Eredeti: {prefix_test}")
    print(f"  Normalizált: {normalize(prefix_test)}")
    print(f"  Eredeti: {alphanum_test_text}")
    # A normalize hívásakor a többi funkció is lefut, ami befolyásolhatja.
    # Ha csak az alfanumerikus részt akarjuk tesztelni, külön kell hívni.
    # De a teljes normalizálás a valós használat.
    print(f"  Normalizált (teljes): {normalize(alphanum_test_text)}")
    # print(f"  Normalizált (csak alfanum): {replace_alphanumeric(alphanum_test_text)}") # Ha csak azt tesztelnénk

    # Dátum teszt
    print("--- Dátum Speciális Teszt ---")
    date_test_text = "2024. jan. 1. és 2024. január 1. valamint 2024. I. 1."
    print(f"  Eredeti: {date_test_text}")
    print(f"  Normalizált (teljes): {normalize(date_test_text)}")
    
    # Római számok tesztelése, amelyek nem alakulnak át, mert nincs utánuk pont
    # és a replace_acronyms sem betűzi őket.
    print("--- Nem átalakuló Római Számok Teszt ---")
    roman_test_text = "XIV Lajos és IV Béla uralkodott. A film címe Mission Impossible III volt."
    print(f"  Eredeti: {roman_test_text}")
    # A "III" itt mozaikszóként lehet, hogy betűzve lesz, ha a replace_acronyms nem elég specifikus.
    # A jelenlegi replace_acronyms `(?!(?:[IVXLCDM]+)\b)` kizárása ezt hivatott megakadályozni.
    print(f"  Normalizált (teljes): {normalize(roman_test_text)}")
    
    # Időpont teszt
    print("--- Időpont Speciális Teszt ---")
    time_test_text = "Találkozó 10:00-kor. Indulás 9:15. Érkezés 11:45:30."
    print(f"  Eredeti: {time_test_text}")
    print(f"  Normalizált (teljes): {normalize(time_test_text)}")

    # Sorszámok és számok együtt
    print("--- Sorszámok és Számok Teszt ---")
    num_ord_test_text = "A 3. fejezet 12 oldalas. 100 emberből a 10. nyert."
    print(f"  Eredeti: {num_ord_test_text}")
    print(f"  Normalizált (teljes): {normalize(num_ord_test_text)}")
    
    # force_changes teszt (szó közben is)
    print("--- Force Changes (szó közben) Teszt ---")
    force_test_text = "Ez egy tw%eet, showly." # % -> százalék, w -> v, ly -> j
    print(f"  Eredeti: {force_test_text}")
    print(f"  Normalizált (teljes): {normalize(force_test_text)}")
