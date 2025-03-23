import csv
import re
from num2words import num2words
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

def load_force_changes(filename="force_changes.csv"):
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
    for key, value in force_changes.items():
        text = text.replace(key, f' {value} ')
    return text

def load_changes(filename="changes.csv"):
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
    for key, value in changes.items():
        pattern = r'\b{}\b'.format(re.escape(key))
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)
    return text

def remove_duplicate_spaces(text):
    # Többszörös szóközök eltávolítása
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def add_prefix(text):
    # stabilize the short predicts with ... frefix
    return '... ' + text.lower()

def normalize(text):
    force_changes = load_force_changes('force_changes.csv')
    changes = load_changes('changes.csv')

    text = apply_force_changes(text, force_changes)
    text = apply_changes(text, changes)
    text = remove_duplicate_spaces(text)
    text = add_prefix(text)

    return text
