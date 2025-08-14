# E2/F5‑TTS REST API – Részletes használati útmutató

*Utolsó frissítés: 2025‑06‑13*

---

## Tartalomjegyzék

1. [Bevezetés](#bevezetés)
2. [Gyors indulás](#gyors-indulás)
3. [Alapfogalmak](#alapfogalmak)
4. [Az API alap URL‑sémája](#az-api-alap-url-sémája)
5. [Végpontok áttekintése](#végpontok-áttekintése)
6. [Kérés felépítése](#kérés-felépítése)
7. [Válasz felépítése](#válasz-felépítése)
8. [Példák](#példák)

   * 8.1. [`basic_tts`](#81-basic_tts) – egyszerű TTS
   * 8.2. [`generate_multistyle_speech`](#82-generate_multistyle_speech) – több hangstílus
   * 8.3. [`switch_tts_model` és `set_custom_model`](#83-model-váltás)
   * 8.4. Python – `gradio_client`-tel
9. [Hibakezelés](#hibakezelés)
10. [Legjobb gyakorlatok](#legjobb-gyakorlatok)
11. [Függelék A – `/config` és `components`](#függelék-a)

---

## Bevezetés

Ez a dokumentum leírja, hogyan használhatod **REST kérésekkel** a Gradio‑alapú E2/F5‑TTS web UI összes fontos funkcióját (modellváltás, szöveg‑felolvasás, multistílusú TTS, chat TTS). A szerver FastAPI‑t használ, de a Gradio 4.x‑től kezdve minden Gradio‑s végpont a **`/gradio_api/…`** útvonal alá kerül.

> **Fontos:** nincs hagyományos `/docs` vagy numerikus `/api/predict/7`; helyette név‑alapú `api_name`‑eket találsz.

---

## Gyors indulás

```bash
# 1 – indítsd a szervert (API alapból engedélyezett)
python app.py -p 7860 --share   # vagy saját flagjeid

# 2 – ellenőrizd, hogy fut-e
curl http://localhost:7860/gradio_api/openapi.json | jq .info

# 3 – listázd a végpontokat és input‑indexeket
curl http://localhost:7860/config | jq '.dependencies[] | {api_name, inputs}'
```

---

## Alapfogalmak

| Fogalom             | Magyarázat                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| **api\_name**       | A callback beszédes neve (pl. `basic_tts`). Ez lesz az URL‑ben.                                        |
| **input‑komponens** | Minden UI elem egy számmal azonosítva (pl. `1725`). A sorrend a `.click(inputs=[…])` listával egyezik. |
| **`data[]`**        | Kérésnél: értékek listája **ugyanabban a sorrendben**, mint az `inputs=[…]`.                           |
| **session\_hash**   | Tetszőleges string, egy munkamenet azonosítására. Ha sorban több hívást küldesz, használd ugyanazt.    |

---

## Az API alap URL‑sémája

```
POST http(s)://<host>:<port>/gradio_api/call/<api_name>
```

* Minden hívás **POST**.
* `Content‑Type: application/json`.
* A törzs két mezőt tartalmaz:

  ```json
  {
    "data": [ /* komponensek sorban */ ],
    "session_hash": "<bármi_azonosító>"
  }
  ```

---

## Végpontok áttekintése

| api\_name                    | Funkció                                                 | Inputs (darab) | Megjegyzés                                |
| ---------------------------- | ------------------------------------------------------- | -------------- | ----------------------------------------- |
| `switch_tts_model`           | Modellválasztó rádiógomb                                | 1              | `"F5-TTS_v1"`, `"E2-TTS"` vagy `"Custom"` |
| `set_custom_model`           | Egyéni modell 3 paramétere                              | 3              | ckpt útvonal, vocab útvonal, JSON cfg     |
| `basic_tts`                  | Egyszerű, egyetlen hangos TTS                           | 10             | l. [8.1](#81-basic_tts)                   |
| `generate_multistyle_speech` | 100 hangstílusig                                        | 249            | l. [8.2](#82-generate_multistyle_speech)  |
| `load_chat_model`            | Chat‑LLM betöltése                                      | 1              | pl. `"Qwen/Qwen2.5-3B-Instruct"`          |
| `chat_pipeline`              | Beszélgetés (ASR→LLM→TTS)                               | 9              | hang v. szöveg, opciók                    |
| …                            | további `load_text_from_file_*`, `insert_…`, `delete_…` | változó        | UI‑segédek, ritkán kell REST‑ből          |

A pontos `inputs` listát mindig nézd meg a `/config` JSON‑ban.

---

## Kérés felépítése

```json
{
  "data": [ <input_1>, <input_2>, … , <input_N> ],
  "session_hash": "egyedi_hex_vagy_uuid"
}
```

* **Típusok**

  * *Fájlnál* – `null` ↔ nincs fájl, vagy **feltöltött URL** (pl. `"file=uploads%2Fabc.wav"`).
  * *Bool* → `true` / `false`.
  * *Audio‑/kép‑output* → base64 URL‑data formátumban érkezik.

---

## Válasz felépítése

```json
{
  "data": [ /* callback return értékei */ ],
  "is_generating": false,
  "duration": 3.18
}
```

* **Audio** → `["audio", "data:audio/wav;base64,UklGR…"]`
* **Kép**   → `"data:image/png;base64,iVBOR…"`

---

## Példák

### 8.1 `basic_tts`

```bash
curl -X POST http://localhost:7860/gradio_api/call/basic_tts \
  -H "Content-Type: application/json" \
  -d '{
        "data": [
          null, "", "Hello from API",    # 0‑2
          false, true, 0,                  # 3‑5
          0.15, 32, 1.0,                   # 6‑8
          "None"                          # 9
        ],
        "session_hash": "tts_demo"
      }' | jq '.data[0]'
```

*Az első elem (`data[0]`) az audio WAV base64‑el; mentheted fájlba.*

### 8.2 `generate_multistyle_speech`

1 kötelező hangstílus (név + audio), minden más `null`/"":

```python
from gradio_client import Client
client = Client("http://localhost:7860")

# adatok összeépítése
payload = [
    '{"name":"Narrator"} Once upon a time…',  # gen_text_input_multistyle
    "Narrator", "file=uploads%2Fvoice.wav", "",  # 1. típus (név, audio, ref_text)
] + [None]*297 + [True, "hun"]  # üres helyek + globális opciók

wav, *_, meta = client.predict(*payload, api_name="/generate_multistyle_speech")
```

### 8.3 Modell váltása

```bash
# F5 → E2
curl -X POST http://localhost:7860/gradio_api/call/switch_tts_model \
  -H "Content-Type: application/json" \
  -d '{"data":["E2-TTS"], "session_hash":"model"}'
```

### 8.4 Python példa `gradio_client`-tel

```python
from gradio_client import Client
client = Client("http://localhost:7860")
wav, spec, _, _ = client.predict(
        None, "", "Szia világ!", False, True, 0, 0.15, 32, 1.0, "hun",
        api_name="/basic_tts")
open("out.wav", "wb").write(wav[1].split(",",1)[1].encode())
```

---

## Hibakezelés

| HTTP kód  | Jelentés                 | Tipikus ok                                         |
| --------- | ------------------------ | -------------------------------------------------- |
| 400 / 422 | Szintaxis vagy típushiba | JSON rossz, `data` méret/típus nem stimmel         |
| 404       | Nincs ilyen `api_name`   | Elírt URL vagy nincs API engedélyezve (`--no-api`) |
| 500       | Belső kivétel            | Modell‑ vagy CUDA‑hiba, hiányzó file               |

A válasz `detail` mezője részletesen kiírja, hogy *Expected X inputs, received Y* stb.

---

## Legjobb gyakorlatok

* Használd a **`gradio_client`** könyvtárat, ha Pythonból hívod → nem kell manuálisan buildelni a `data` listát.
* Több kérésnél reutilizáld a **`session_hash`**‑t → gyorsabb load, stabil queue.
* **`null`** minden nem kötelező fájlhelyen; ne üres string.
* Hosszú hívásoknál (TTS > 30 s) számíts sorban állásra (`queue_join`).
* Base64‑es audio nagy; ha csak tesztelsz, küldj `Accept: application/json`‑t (ez az alap).

---

## Függelék A – `/config` és `components`

A `GET /config` válasza:

```json
{
  "version": "4.24.0",
  "mode": "grpc",
  "components": { "4": {"type":"file", "label":"Reference Audio"}, … },
  "dependencies": [
    { "api_name": "basic_tts", "inputs": [4,1725,6,…] },
    …
  ]
}
```

* A `components` szekcióban látszik, milyen **típus** (`file`, `textbox`, `slider`, …).
* A `dependencies[*].inputs` lista **sorrendje** → ez határozza meg a `data[]` sorrendet.

---

**Ennyi!** Ezzel a leírással önállóan is tudod scriptből vezérelni a TTS‑szervert, modellváltástól a multistílusú hangszintézisig.

