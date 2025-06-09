from normaliser import normalize

# Tesztesetek
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

# Teljes tesztfolyamat
def run_tests():
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

    # Speciális tesztek
    print("--- Alfanumerikus Speciális Teszt ---")
    alphanum_test_text = "A1, B2B, C3C3, D4D4D, E5-F6, G7H8I9, J10K"
    print(f"  Eredeti: {alphanum_test_text}")
    print(f"  Normalizált (teljes): {normalize(alphanum_test_text)}")
    
    print("--- Szöveg elejének ellenőrzése ---")
    prefix_test = "   Ez egy szöveg szóközökkel."
    print(f"  Eredeti: {prefix_test}")
    print(f"  Normalizált: {normalize(prefix_test)}")
    
    print("--- Dátum Speciális Teszt ---")
    date_test_text = "2024. jan. 1. és 2024. január 1. valamint 2024. I. 1."
    print(f"  Eredeti: {date_test_text}")
    print(f"  Normalizált (teljes): {normalize(date_test_text)}")
    
    print("--- Nem átalakuló Római Számok Teszt ---")
    roman_test_text = "XIV Lajos és IV Béla uralkodott. A film címe Mission Impossible III volt."
    print(f"  Eredeti: {roman_test_text}")
    print(f"  Normalizált (teljes): {normalize(roman_test_text)}")
    
    print("--- Időpont Speciális Teszt ---")
    time_test_text = "Találkozó 10:00-kor. Indulás 9:15. Érkezés 11:45:30."
    print(f"  Eredeti: {time_test_text}")
    print(f"  Normalizált (teljes): {normalize(time_test_text)}")

    print("--- Sorszámok és Számok Teszt ---")
    num_ord_test_text = "A 3. fejezet 12 oldalas. 100 emberből a 10. nyert."
    print(f"  Eredeti: {num_ord_test_text}")
    print(f"  Normalizált (teljes): {normalize(num_ord_test_text)}")
    
    print("--- Force Changes (szó közben) Teszt ---")
    force_test_text = "Ez egy tw%eet, showly." # % -> százalék, w -> v, ly -> j
    print(f"  Eredeti: {force_test_text}")
    print(f"  Normalizált (teljes): {normalize(force_test_text)}")

if __name__ == "__main__":
    run_tests()
