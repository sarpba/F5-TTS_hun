# ruff: noqa: E402
import json
import re
import tempfile
import os  # szükséges a normaliser mappák beolvasásához
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


def get_normaliser_choices():
    normaliser_path = "./normalisers"
    choices = []
    if os.path.exists(normaliser_path) and os.path.isdir(normaliser_path):
        for item in os.listdir(normaliser_path):
            subdir = os.path.join(normaliser_path, item)
            if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "normaliser.py")):
                choices.append(item)
    return choices

normaliser_choices = get_normaliser_choices()
all_tts_choices = [DEFAULT_TTS_MODEL, "E2-TTS", "Custom"]

# load models
vocoder = load_vocoder()


def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_e2tts():
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)


def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)


F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None


@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
            pre_custom_path = model[1]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence if enabled
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text


@gpu_decorator
def basic_tts(
    ref_audio_input,
    ref_text_input,
    gen_text_input,
    remove_silence,
    cross_fade_duration_slider,
    nfe_slider,
    speed_slider,
    normaliser_choice_input,  # A Choose Normaliser komponens értéke
):
    if normaliser_choice_input != "None":
        normaliser_file = os.path.join("normalisers", normaliser_choice_input, "normaliser.py")
        if os.path.exists(normaliser_file):
            import importlib.util
            spec = importlib.util.spec_from_file_location("normaliser", normaliser_file)
            normaliser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(normaliser_module)
            if hasattr(normaliser_module, "normalize"):
                processed_text = normaliser_module.normalize(gen_text_input)
            else:
                print("A normaliser.py nem tartalmazza a 'normalize' függvényt, az eredeti szöveg kerül használatra.")
                processed_text = gen_text_input
        else:
            print("A megadott normaliser.py fájl nem található, az eredeti szöveg kerül használatra.")
            processed_text = gen_text_input
    else:
        processed_text = gen_text_input

    actual_model = tts_model_choice
    audio_out, spectrogram_path, ref_text_out = infer(
        ref_audio_input,
        ref_text_input,
        processed_text,
        actual_model,
        remove_silence,
        cross_fade_duration=cross_fade_duration_slider,
        nfe_step=nfe_slider,
        speed=speed_slider,
    )
    return audio_out, spectrogram_path, ref_text_out


@gpu_decorator
def generate_multistyle_speech(
    gen_text,
    *args,
):
    # Várható argok:
    # args[0 : max_speech_types]         -> speech_type_names
    # args[max_speech_types : 2*max_speech_types] -> speech_type_audios
    # args[2*max_speech_types : 3*max_speech_types] -> speech_type_ref_texts
    # args[3*max_speech_types]           -> remove_silence_multistyle
    # args[3*max_speech_types + 1]       -> global normaliser választás
    max_speech_types = 100
    speech_type_names_list = args[:max_speech_types]
    speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
    speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
    remove_silence = args[3 * max_speech_types]
    normaliser_choice = args[-1]  # utolsó elem
    # Ha van normaliser, próbáljuk meg betölteni a modult egyszer:
    if normaliser_choice != "None":
        normaliser_file = os.path.join("normalisers", normaliser_choice, "normaliser.py")
        if os.path.exists(normaliser_file):
            import importlib.util
            spec = importlib.util.spec_from_file_location("normaliser", normaliser_file)
            normaliser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(normaliser_module)
        else:
            normaliser_module = None
    else:
        normaliser_module = None

    speech_types = OrderedDict()
    ref_text_idx = 0
    for name_input, audio_input, ref_text_input in zip(
        speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
    ):
        if name_input and audio_input:
            speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
        else:
            speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
        ref_text_idx += 1

    # Parse the input text into segments; a sorok között {style} megjelölés található
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_style = "Regular"
    for i in range(len(tokens)):
        if i % 2 == 0:
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            current_style = tokens[i].strip()

    generated_audio_segments = []
    for segment in segments:
        style = segment["style"]
        text = segment["text"]
        # Alapértelmezett: ha a megadott stílus nincs definiálva, akkor Regular
        if style in speech_types:
            current_style = style
        else:
            gr.Warning(f"Type {style} is not available, using Regular as default.")
            current_style = "Regular"
        # Ha van normaliser, normalizáljuk a sor szövegét
        if normaliser_module is not None and hasattr(normaliser_module, "normalize"):
            text = normaliser_module.normalize(text)
        try:
            ref_audio = speech_types[current_style]["audio"]
        except KeyError:
            gr.Warning(f"Please provide reference audio for type {current_style}.")
            return [None] + [speech_types[style]["ref_text"] for style in speech_types]
        ref_text = speech_types[current_style].get("ref_text", "")
        audio_out, _, ref_text_out = infer(
            ref_audio, ref_text, text, tts_model_choice, remove_silence, 0, show_info=print
        )
        sr, audio_data = audio_out
        generated_audio_segments.append(audio_data)
        speech_types[current_style]["ref_text"] = ref_text_out
    if generated_audio_segments:
        final_audio_data = np.concatenate(generated_audio_segments)
        return [(sr, final_audio_data)] + [speech_types[style]["ref_text"] for style in speech_types]
    else:
        gr.Warning("No audio generated.")
        return [None] + [speech_types[style]["ref_text"] for style in speech_types]


@gpu_decorator
def process_audio_input(audio_path, text, history, conv_state):
    if not audio_path and not text.strip():
        return history, conv_state, ""
    if audio_path:
        text = preprocess_ref_audio_text(audio_path, text)[1]
    if not text.strip():
        return history, conv_state, ""
    conv_state.append({"role": "user", "content": text})
    history.append((text, None))
    response = generate_response(conv_state, chat_model_state, chat_tokenizer_state)
    conv_state.append({"role": "assistant", "content": response})
    history[-1] = (text, response)
    return history, conv_state, ""


@gpu_decorator
def generate_audio_response(history, ref_audio, ref_text, remove_silence):
    if not history or not ref_audio:
        return None
    last_user_message, last_ai_response = history[-1]
    if not last_ai_response:
        return None
    audio_result, _, ref_text_out = infer(
        ref_audio,
        ref_text,
        last_ai_response,
        tts_model_choice,
        remove_silence,
        cross_fade_duration=0.15,
        speed=1.0,
        show_info=print,
    )
    return audio_result, ref_text_out


def clear_conversation():
    return [], [
        {
            "role": "system",
            "content": "You are not an AI assistant, you are whoever the user says you are. Keep your responses concise.",
        }
    ]


def update_system_prompt(new_prompt):
    new_conv_state = [{"role": "system", "content": new_prompt}]
    return [], new_conv_state


# ======================
# UI: Egyetlen fő app blokkban
# ======================
with gr.Blocks() as app:
    gr.Markdown(
        """
# E2/F5 TTS

Ez egy helyi web UI az F5-TTS (és E2-TTS) modellhez, amely fejlett batch feldolgozást támogat.
A következő TTS modellek közül választhatsz:
* [F5-TTS](https://arxiv.org/abs/2410.06885)
* [E2-TTS](https://arxiv.org/abs/2406.18009)

Ha problémád van, próbáld meg a referens audio WAV/MP3 formátumba konvertálni, vagy 12 másodpercre vágni.
**MEGJEGYZÉS:** A referens szöveget automatikusan transzkripcióval állítjuk elő, ha nem adod meg.
"""
    )
    # TTS modell választó sor
    with gr.Row():
        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=all_tts_choices,
                label="Choose TTS Model",
                value=DEFAULT_TTS_MODEL,
            )
        else:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS"],
                label="Choose TTS Model",
                value=DEFAULT_TTS_MODEL,
            )
        # Custom model komponensek (ha szükséges)
        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=DEFAULT_TTS_MODEL_CFG[0],
            allow_custom_value=True,
            label="Model: local_path | hf://user_id/repo_id/model_ckpt",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=DEFAULT_TTS_MODEL_CFG[1],
            allow_custom_value=True,
            label="Vocab: local_path | hf://user_id/repo_id/vocab_file",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(
                    dict(
                        dim=1024,
                        depth=22,
                        heads=16,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
                json.dumps(
                    dict(
                        dim=768,
                        depth=18,
                        heads=12,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
            ],
            value=DEFAULT_TTS_MODEL_CFG[2],
            allow_custom_value=True,
            label="Config: in a dictionary form",
            visible=False,
        )
    # Choose Normaliser komponens – globális, a Basic-TTS és a Multi-Speech számára egyaránt
    with gr.Row():
        global_choose_normaliser = gr.Radio(
            choices=["None"] + get_normaliser_choices(),
            label="Choose Normaliser",
            value="None",
        )
    # TabbedInterface a fő UI alsó részén
    tab_basic_tts = gr.Blocks()
    with tab_basic_tts:
        with gr.Accordion("Basic-TTS", open=True):
            gr.Markdown("### Batched TTS")
            ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
            gen_text_input = gr.Textbox(label="Text to Generate", lines=10)
            generate_btn = gr.Button("Synthesize", variant="primary")
            with gr.Accordion("Advanced Settings", open=False):
                ref_text_input = gr.Textbox(
                    label="Reference Text",
                    info="Ha megadod, nem történik automatikus transzkripció.",
                    lines=2,
                )
                remove_silence = gr.Checkbox(
                    label="Remove Silences",
                    info="A hosszabb audio esetén a modell csendeket is generál.",
                    value=False,
                )
                speed_slider = gr.Slider(
                    label="Speed",
                    minimum=0.3,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    info="A hang sebességének beállítása.",
                )
                nfe_slider = gr.Slider(
                    label="NFE Steps",
                    minimum=4,
                    maximum=64,
                    value=32,
                    step=2,
                    info="A denoising lépések száma.",
                )
                cross_fade_duration_slider = gr.Slider(
                    label="Cross-Fade Duration (s)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.15,
                    step=0.01,
                    info="Az audio klip-ek közötti átfedés ideje.",
                )
            audio_output = gr.Audio(label="Synthesized Audio")
            spectrogram_output = gr.Image(label="Spectrogram")
            generate_btn.click(
                basic_tts,
                inputs=[
                    ref_audio_input,
                    ref_text_input,
                    gen_text_input,
                    remove_silence,
                    cross_fade_duration_slider,
                    nfe_slider,
                    speed_slider,
                    global_choose_normaliser,
                ],
                outputs=[audio_output, spectrogram_output, ref_text_input],
            )
    tab_multistyle = gr.Blocks()
    with tab_multistyle:
        with gr.Accordion("Multi-Speech", open=True):
            gr.Markdown("### Multiple Speech-Type Generation")
            gr.Markdown("**Példa bemenet:** {Regular} Hello... {Surprised} ...")
            with gr.Row() as regular_row:
                with gr.Column():
                    regular_name = gr.Textbox(value="Regular", label="Speech Type Name")
                    regular_insert = gr.Button("Insert Label", variant="secondary")
                regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")
                regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=2)
            max_speech_types = 100
            speech_type_rows = [regular_row]
            speech_type_names = [regular_name]
            speech_type_audios = [regular_audio]
            speech_type_ref_texts = [regular_ref_text]
            speech_type_delete_btns = [None]
            speech_type_insert_btns = [regular_insert]
            for i in range(max_speech_types - 1):
                with gr.Row(visible=False) as row:
                    with gr.Column():
                        name_input = gr.Textbox(label="Speech Type Name")
                        delete_btn = gr.Button("Delete Type", variant="secondary")
                        insert_btn = gr.Button("Insert Label", variant="secondary")
                    audio_input = gr.Audio(label="Reference Audio", type="filepath")
                    ref_text_input_ms = gr.Textbox(label="Reference Text", lines=2)
                speech_type_rows.append(row)
                speech_type_names.append(name_input)
                speech_type_audios.append(audio_input)
                speech_type_ref_texts.append(ref_text_input_ms)
                speech_type_delete_btns.append(delete_btn)
                speech_type_insert_btns.append(insert_btn)
            add_speech_type_btn = gr.Button("Add Speech Type")
            # speech_type_count tárolása listában
            speech_type_count = [1]
            def add_speech_type_fn():
                row_updates = [gr.update() for _ in range(max_speech_types)]
                if speech_type_count[0] < max_speech_types:
                    row_updates[speech_type_count[0]] = gr.update(visible=True)
                    speech_type_count[0] += 1
                else:
                    gr.Warning("Elérted a maximum speech type számot.")
                return row_updates
            add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)
            def delete_speech_type_fn():
                return gr.update(visible=False), None, None, None
            for i in range(1, len(speech_type_delete_btns)):
                speech_type_delete_btns[i].click(
                    delete_speech_type_fn,
                    outputs=[speech_type_rows[i], speech_type_names[i], speech_type_audios[i], speech_type_ref_texts[i]],
                )
            gen_text_input_multistyle = gr.Textbox(
                label="Text to Generate",
                lines=10,
                placeholder="Írd be a szkriptet, speaker/emóció megjelöléssel..."
            )
            def make_insert_speech_type_fn(index):
                def insert_speech_type_fn(current_text, speech_type_name):
                    current_text = current_text or ""
                    speech_type_name = speech_type_name or "None"
                    updated_text = current_text + f"{{{speech_type_name}}} "
                    return updated_text
                return insert_speech_type_fn
            for i, insert_btn in enumerate(speech_type_insert_btns):
                insert_fn = make_insert_speech_type_fn(i)
                insert_btn.click(
                    insert_fn,
                    inputs=[gen_text_input_multistyle, speech_type_names[i]],
                    outputs=gen_text_input_multistyle,
                )
            with gr.Accordion("Advanced Settings", open=False):
                remove_silence_multistyle = gr.Checkbox(
                    label="Remove Silences",
                    value=True,
                )
            generate_multistyle_btn = gr.Button("Generate Multi-Style Speech", variant="primary")
            audio_output_multistyle = gr.Audio(label="Synthesized Audio")
            # Fontos: a generate_multistyle_speech input listájához hozzáadjuk a global_choose_normaliser-t is!
            generate_multistyle_btn.click(
                generate_multistyle_speech,
                inputs=[gen_text_input_multistyle] + speech_type_names + speech_type_audios + speech_type_ref_texts + [remove_silence_multistyle, global_choose_normaliser],
                outputs=[audio_output_multistyle] + speech_type_ref_texts,
            )
            def validate_speech_types(gen_text, regular_name, *args):
                speech_type_names_list = args
                speech_types_available = set()
                if regular_name:
                    speech_types_available.add(regular_name)
                for name_input in speech_type_names_list:
                    if name_input:
                        speech_types_available.add(name_input)
                pattern = r"\{(.*?)\}"
                tokens = re.split(pattern, gen_text)
                segments = []
                current_style = "Regular"
                for i in range(len(tokens)):
                    if i % 2 == 0:
                        text = tokens[i].strip()
                        if text:
                            segments.append({"style": current_style, "text": text})
                    else:
                        current_style = tokens[i].strip()
                speech_types_in_text = set(segment["style"] for segment in segments)
                missing_speech_types = speech_types_in_text - speech_types_available
                if missing_speech_types:
                    return gr.update(interactive=False)
                else:
                    return gr.update(interactive=True)
            gen_text_input_multistyle.change(
                validate_speech_types,
                inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
                outputs=generate_multistyle_btn,
            )
    tab_voice_chat = gr.Blocks()
    with tab_voice_chat:
        with gr.Accordion("Voice-Chat", open=True):
            gr.Markdown("### Voice Chat\nBeszélgetés az AI-vel a referens hangoddal.")
            if not USING_SPACES:
                load_chat_model_btn = gr.Button("Load Chat Model", variant="primary")
                chat_interface_container = gr.Column(visible=False)
                @gpu_decorator
                def load_chat_model():
                    global chat_model_state, chat_tokenizer_state
                    if chat_model_state is None:
                        show_info = gr.Info
                        show_info("Loading chat model...")
                        model_name = "Qwen/Qwen2.5-3B-Instruct"
                        chat_model_state = AutoModelForCausalLM.from_pretrained(
                            model_name, torch_dtype="auto", device_map="auto"
                        )
                        chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)
                        show_info("Chat model loaded.")
                    return gr.update(visible=False), gr.update(visible=True)
                load_chat_model_btn.click(load_chat_model, outputs=[load_chat_model_btn, chat_interface_container])
            else:
                chat_interface_container = gr.Column()
                if chat_model_state is None:
                    model_name = "Qwen/Qwen2.5-3B-Instruct"
                    chat_model_state = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
                    chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)
            with chat_interface_container:
                with gr.Row():
                    with gr.Column():
                        ref_audio_chat = gr.Audio(label="Reference Audio", type="filepath")
                    with gr.Column():
                        with gr.Accordion("Advanced Settings", open=False):
                            remove_silence_chat = gr.Checkbox(
                                label="Remove Silences",
                                value=True,
                            )
                            ref_text_chat = gr.Textbox(
                                label="Reference Text",
                                info="Opcionális: hagyd üresen, ha auto-transzkripciót szeretnél",
                                lines=2,
                            )
                            system_prompt_chat = gr.Textbox(
                                label="System Prompt",
                                value="You are not an AI assistant, you are whoever the user says you are. Keep your responses concise.",
                                lines=2,
                            )
                chatbot_interface = gr.Chatbot(label="Conversation")
                with gr.Row():
                    with gr.Column():
                        audio_input_chat = gr.Microphone(label="Speak your message", type="filepath")
                        audio_output_chat = gr.Audio(autoplay=True)
                    with gr.Column():
                        text_input_chat = gr.Textbox(label="Type your message", lines=1)
                        send_btn_chat = gr.Button("Send Message")
                        clear_btn_chat = gr.Button("Clear Conversation")
                conversation_state = gr.State(
                    value=[
                        {
                            "role": "system",
                            "content": "You are not an AI assistant, you are whoever the user says you are. Keep your responses concise.",
                        }
                    ]
                )
                audio_input_chat.stop_recording(
                    process_audio_input,
                    inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
                    outputs=[chatbot_interface, conversation_state],
                ).then(
                    generate_audio_response,
                    inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
                    outputs=[audio_output_chat, ref_text_chat],
                ).then(
                    lambda: None,
                    None,
                    audio_input_chat,
                )
                text_input_chat.submit(
                    process_audio_input,
                    inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
                    outputs=[chatbot_interface, conversation_state],
                ).then(
                    generate_audio_response,
                    inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
                    outputs=[audio_output_chat, ref_text_chat],
                ).then(
                    lambda: None,
                    None,
                    text_input_chat,
                )
                send_btn_chat.click(
                    process_audio_input,
                    inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
                    outputs=[chatbot_interface, conversation_state],
                ).then(
                    generate_audio_response,
                    inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
                    outputs=[audio_output_chat, ref_text_chat],
                ).then(
                    lambda: None,
                    None,
                    text_input_chat,
                )
                clear_btn_chat.click(
                    clear_conversation,
                    outputs=[chatbot_interface, conversation_state],
                )
                system_prompt_chat.change(
                    update_system_prompt,
                    inputs=system_prompt_chat,
                    outputs=[chatbot_interface, conversation_state],
                )
    tab_credits = gr.Blocks()
    with tab_credits:
        with gr.Accordion("Credits", open=True):
            gr.Markdown(
                """
# Credits

* [mrfakename](https://github.com/fakerybakery) – az eredeti online demoért.
* [RootingInLoad](https://github.com/RootingInLoad) – chunk generation és podcast app ötletért.
* [jpgallegoar](https://github.com/jpgallegoar) – multi-speech és voice chat funkciókért.
"""
            )
    gr.TabbedInterface(
        [tab_basic_tts, tab_multistyle, tab_voice_chat, tab_credits],
        ["Basic-TTS", "Multi-Speech", "Voice-Chat", "Credits"],
    )
    # --- TTS modell váltás kezelése ---
    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")
    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG
    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "Custom":
            custom_ckpt_path_update = gr.update(visible=True, value=load_last_used_custom()[0])
            custom_vocab_path_update = gr.update(visible=True, value=load_last_used_custom()[1])
            custom_model_cfg_update = gr.update(visible=True, value=load_last_used_custom()[2])
            tts_model_choice = ["Custom", load_last_used_custom()[0], load_last_used_custom()[1], json.loads(load_last_used_custom()[2])]
            return custom_ckpt_path_update, custom_vocab_path_update, custom_model_cfg_update
        else:
            tts_model_choice = new_choice
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")
    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

if __name__ == "__main__":
    @click.command()
    @click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
    @click.option("--host", "-H", default=None, help="Host to run the app on")
    @click.option("--share", "-s", default=False, is_flag=True, help="Share the app via Gradio share link")
    @click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
    @click.option("--root_path", "-r", default=None, type=str, help="The root path of the application")
    @click.option("--inbrowser", "-i", is_flag=True, default=False, help="Automatically launch the interface in the default web browser")
    def main(port, host, share, api, root_path, inbrowser):
        print("Starting app...")
        app.queue(api_open=api).launch(
            server_name=host,
            server_port=port,
            share=share,
            show_api=api,
            root_path=root_path,
            inbrowser=inbrowser,
        )
    main()
