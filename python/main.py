import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import soundfile
import onnxruntime as ort
import axengine as axe
import argparse
import time
from split_utils import split_sentence
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from symbols import LANG_TO_SYMBOL_MAP
import re

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def get_text_for_tts_infer(text, language_str, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    phone = intersperse(phone, 0)
    tone = intersperse(tone, 0)
    language = intersperse(language, 0)

    phone = np.array(phone, dtype=np.int32)
    tone = np.array(tone, dtype=np.int32)
    language = np.array(language, dtype=np.int32)
    word2ph = np.array(word2ph, dtype=np.int32) * 2
    word2ph[0] += 1

    return phone, tone, language, norm_text, word2ph

def split_sentences_into_pieces(text, language, quiet=False):
    texts = split_sentence(text, language_str=language)
    if not quiet:
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
    return texts

def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments

def merge_sub_audio(sub_audio_list, pad_size, audio_len):
    if pad_size > 0:
        for i in range(len(sub_audio_list) - 1):
            sub_audio_list[i][-pad_size:] += sub_audio_list[i+1][:pad_size]
            sub_audio_list[i][-pad_size:] /= 2
            if i > 0:
                sub_audio_list[i] = sub_audio_list[i][pad_size:]

    sub_audio = np.concatenate(sub_audio_list, axis=-1)
    return sub_audio[:audio_len]

def calc_word2pronoun(word2ph, pronoun_lens):
    indice = [0]
    for ph in word2ph[:-1]:
        indice.append(indice[-1] + ph)
    word2pronoun = []
    for i, ph in zip(indice, word2ph):
        word2pronoun.append(np.sum(pronoun_lens[i : i + ph]))
    return word2pronoun

def generate_slices(word2pronoun, dec_len):
    pn_start, pn_end = 0, 0
    zp_start, zp_end = 0, 0
    zp_len = 0
    pn_slices = []
    zp_slices = []
    while pn_end < len(word2pronoun):
        if pn_end - pn_start > 2 and np.sum(word2pronoun[pn_end - 2 : pn_end + 1]) <= dec_len:
            zp_len = np.sum(word2pronoun[pn_end - 2 : pn_end])
            zp_start = zp_end - zp_len
            pn_start = pn_end - 2
        else:
            zp_len = 0
            zp_start = zp_end
            pn_start = pn_end
            
        while pn_end < len(word2pronoun) and zp_len + word2pronoun[pn_end] <= dec_len:
            zp_len += word2pronoun[pn_end]
            pn_end += 1
        zp_end = zp_start + zp_len
        pn_slices.append(slice(pn_start, pn_end))
        zp_slices.append(slice(zp_start, zp_end))
    return pn_slices, zp_slices

def load_models(args):
    if args.encoder is None:
        if "ZH" in args.language:
            enc_model = "../models/encoder-zh.onnx"
        else:
            enc_model = f"../models/encoder-{args.language.lower()}.onnx"
        assert os.path.exists(enc_model), f"Encoder model ({enc_model}) not exist!"
    else:
        enc_model = args.encoder

    if args.decoder is None:
        if "ZH" in args.language:
            dec_model = "../models/decoder-zh.axmodel"
        else:
            dec_model = f"../models/decoder-{args.language.lower()}.axmodel"
        assert os.path.exists(dec_model), f"Decoder model ({dec_model}) not exist!"
    else:
        dec_model = args.decoder

    print(f"Loading models:\n encoder: {enc_model}\n decoder: {dec_model}")

    start = time.time()
    sess_enc = ort.InferenceSession(enc_model, providers=["CPUExecutionProvider"])
    sess_dec = axe.InferenceSession(dec_model)
    g = np.fromfile(f"../models/g-{args.language.lower()}.bin", dtype=np.float32).reshape(1, 256, 1)
    print(f"Models loaded in {1000*(time.time()-start):.1f}ms")
    
    return sess_enc, sess_dec, g

def run_inference(sess_enc, sess_dec, g, args, text):
    _symbol_to_id = {s: i for i, s in enumerate(LANG_TO_SYMBOL_MAP[args.language])}
    audio_list = []
    
    sens = split_sentences_into_pieces(text, args.language)
    
    for n, se in enumerate(sens):
        if args.language in ['EN', 'ZH_MIX_EN']:
            se = re.sub(r'([a-z])([A-Z])', r'\1 \2', se)
            
        phones, tones, lang_ids, _, word2ph = get_text_for_tts_infer(se, args.language, _symbol_to_id)
        
        start = time.time()
        z_p, pronoun_lens, audio_len = sess_enc.run(
            None,
            input_feed={
                'phone': phones,
                'g': g,
                'tone': tones,
                'language': lang_ids,
                'noise_scale': np.array([0.6], dtype=np.float32),
                'length_scale': np.array([1.0/args.speed], dtype=np.float32),
                'noise_scale_w': np.array([0.8], dtype=np.float32),
                'sdp_ratio': np.array([0.2], dtype=np.float32)
            }
        )
        print(f"Encoder inference: {1000*(time.time()-start):.1f}ms")

        word2pronoun = calc_word2pronoun(word2ph, pronoun_lens)
        pn_slices, zp_slices = generate_slices(word2pronoun, args.dec_len)
        
        sub_audio_list = []
        for i, (ps, zs) in enumerate(zip(pn_slices, zp_slices)):
            zp_slice = z_p[..., zs]

            sub_dec_len = zp_slice.shape[-1]
            sub_audio_len = 512 * sub_dec_len
            
            if zp_slice.shape[-1] < args.dec_len:
                zp_slice = np.concatenate(
                    (zp_slice, np.zeros((*zp_slice.shape[:-1], args.dec_len - zp_slice.shape[-1]), dtype=np.float32)),
                    axis=-1
                )
            
            start = time.time()
            audio = sess_dec.run(None, input_feed={"z_p": zp_slice, "g": g})[0].flatten()
            print(f"Decoder slice[{i}]: {1000*(time.time()-start):.1f}ms")
            
            audio_start = 0
            if len(sub_audio_list) > 0:
                if pn_slices[i - 1].stop > ps.start:
                    audio_start = 512 * word2pronoun[ps.start]
    
            audio_end = sub_audio_len
            if i < len(pn_slices) - 1:
                if ps.stop > pn_slices[i + 1].start:
                    audio_end = sub_audio_len - 512 * word2pronoun[ps.stop - 1]

            sub_audio_list.append(audio[audio_start:audio_end])
        
        audio_list.append(merge_sub_audio(sub_audio_list, 0, audio_len[0]))
    
    return audio_numpy_concat(audio_list, args.sample_rate, args.speed)

def get_args():
    parser = argparse.ArgumentParser(
        prog="melotts",
        description="Interactive TTS with one-time model initialization"
    )
    parser.add_argument("--wav_base", "-w", type=str, default="output",
                        help="Base filename for output WAV files")
    parser.add_argument("--encoder", "-e", type=str, default=None)
    parser.add_argument("--decoder", "-d", type=str, default=None)
    parser.add_argument("--dec_len", type=int, default=128)
    parser.add_argument("--sample_rate", "-sr", type=int, default=44100)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--language", "-l", type=str, 
                        choices=["ZH", "ZH_MIX_EN", "JP", "EN", 'KR', "ES", "SP","FR"], 
                        default="ZH_MIX_EN")
    return parser.parse_args()

def main():
    args = get_args()
    output_dir = os.path.dirname(args.wav_base)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    sess_enc, sess_dec, g = load_models(args)
    file_counter = 1

    try:
        while True:
            text = input("\n请输入要合成的文本（输入'exit'退出）: ").strip()
            if text.lower() in ('exit', 'quit'):
                break
            if not text:
                print("输入不能为空！")
                continue

            output_path = f"{args.wav_base}_{file_counter}.wav"
            final_audio = run_inference(sess_enc, sess_dec, g, args, text)
            soundfile.write(output_path, final_audio, args.sample_rate)
            print(f"✓ 音频已保存至: {os.path.abspath(output_path)}")
            file_counter += 1

    except KeyboardInterrupt:
        print("\n操作已中断")
    finally:
        print("感谢使用，再见！")

if __name__ == "__main__":
    main()
