# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "style-bert-vits2==2.5.0",
#   "pyopenjtalk==0.4.1",
#   "numpy<2.0",
#   "torch<2.4",
#   "torchaudio<2.4",
#   "ctranslate2==4.6.3",
#   "transformers==4.57.3",
#   "librosa==0.11.0",
# ]
# ///

import sys,os
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import wave

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
warnings.simplefilter("ignore")
warnings.filterwarnings(
    "ignore",
    message=r"MPS: The constant padding of more than 3 dimensions is not currently supported natively\..*",
    category=UserWarning,
)
# style-bert-vits2のログを設定
import loguru
loguru.logger.remove()  # 既存のログ設定を削除
loguru.logger.add(sys.stderr, level="ERROR")  # ERRORレベルのログのみを表示

import numpy as np
from numpy.typing import NDArray
import torch
import librosa

from style_bert_vits2.constants import DEFAULT_BERT_TOKENIZER_PATHS, Languages, DEFAULT_STYLE
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel as SBV2_TTSModel

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

def download_hf_hub(repo_id: str, path: str|None=None, *, subfolder:str|None=None, cache_dir: str|Path|None=None) -> str:
    # Hugging Face Hubからモデルをダウンロード
    for b in (True, False):
        try:
            if path:
                model_path = hf_hub_download(
                    repo_id=repo_id, filename=path,
                    subfolder=subfolder, cache_dir=cache_dir,
                    local_files_only=b,
                )
            else:
                model_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir, local_files_only=b,
                )
            return model_path
        except LocalEntryNotFoundError as e:
            pass
        except Exception as e:
            raise e
    raise FileNotFoundError(f"{repo_id} {subfolder} {path} not found")


# 言語ごとのデフォルトの BERT トークナイザーのhugginfaceのパス
# .cache/huggingface/hub/に保存されるはず
SBV2_TOKENIZER_PATHS = {
    Languages.JP: f"ku-nlp/{DEFAULT_BERT_TOKENIZER_PATHS[Languages.JP].name}",
    Languages.EN: f"microsoft/{DEFAULT_BERT_TOKENIZER_PATHS[Languages.EN].name}",
    Languages.ZH: f"hfl/{DEFAULT_BERT_TOKENIZER_PATHS[Languages.ZH].name}",
}
SBV2_MODELS = {
    'koharune-ami': {
        'model':{
            'repo_id':'litagin/sbv2_koharune_ami',
            'path':'koharune-ami/koharune-ami.safetensors'
            },
        'config': {
            'repo_id':'litagin/sbv2_koharune_ami',
            'path': 'koharune-ami/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/sbv2_koharune_ami',
            'path': 'koharune-ami/style_vectors.npy'
            },
        'language': 'jp',
        'model_name': '小春音アミ',
        'gender': 'female',
        'license': 'あみたろの声素材工房規約（配布元規約準拠）',
        'license_url': 'https://amitaro.net/voice/voice_rule/',
        'usage_terms': 'amitaro.netの規約（voice_rule と livevoice）を遵守。年齢制限用途・政治/宗教/マルチ・誹謗中傷用途は禁止。公開時は「あみたろの声素材工房 (https://amitaro.net/)」のクレジット表記が必要。',
    },
    "amitaro": {
        'model':{
            'repo_id':'litagin/sbv2_amitaro',
            'path':'amitaro/amitaro.safetensors'
            },
        'config': {
            'repo_id':'litagin/sbv2_amitaro',
            'path': 'amitaro/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/sbv2_amitaro',
            'path': 'amitaro/style_vectors.npy'
            },
        'language': 'jp',
        'model_name': 'あみたろ',
        'gender': 'female',
        'license': 'あみたろの声素材工房規約（配布元規約準拠）',
        'license_url': 'https://amitaro.net/voice/voice_rule/',
        'usage_terms': 'amitaro.netの規約（voice_rule と livevoice）を遵守。年齢制限用途・政治/宗教/マルチ・誹謗中傷用途は禁止。公開時は「あみたろの声素材工房 (https://amitaro.net/)」のクレジット表記が必要。',
    },
    'jvnv-F1-jp': {
        'model':{
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path':'jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors'
            },
        'config': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-F1-jp/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-F1-jp/style_vectors.npy'
            },
        'language': 'jp',
        'model_name': 'JVNV F1',
        'description': 'JVNVコーパス女性話者1',
        'gender': 'female',
        'license': 'CC BY-SA 4.0（JVNVコーパス継承）',
        'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/deed.ja',
        'usage_terms': 'JVNVコーパス由来のためCC BY-SA 4.0を継承。表示・継承条件を満たして利用してください。',
    },
    'jvnv-F2-jp': {
        'model':{
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path':'jvnv-F2-jp/jvnv-F2_e166_s20000.safetensors'
            },
        'config': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-F2-jp/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-F2-jp/style_vectors.npy'
            },
        'language': 'jp',
        'model_name': 'JVNV F2',
        'description': 'JVNVコーパス女性話者2',
        'gender': 'female',
        'license': 'CC BY-SA 4.0（JVNVコーパス継承）',
        'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/deed.ja',
        'usage_terms': 'JVNVコーパス由来のためCC BY-SA 4.0を継承。表示・継承条件を満たして利用してください。',
    },
    'jvnv-M1-jp': {
        'model':{
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path':'jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors'
            },
        'config': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-M1-jp/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-M1-jp/style_vectors.npy'
            },
        'language': 'jp',
        'model_name': 'JVNV M1',
        'gender': 'male',
        'description': 'JVNVコーパス男性話者1',
        'license': 'CC BY-SA 4.0（JVNVコーパス継承）',
        'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/deed.ja',
        'usage_terms': 'JVNVコーパス由来のためCC BY-SA 4.0を継承。表示・継承条件を満たして利用してください。',
    },
    'jvnv-M2-jp': {
        'model':{
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path':'jvnv-M2-jp/jvnv-M2-jp_e159_s17000.safetensors'
            },
        'config': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-M2-jp/config.json'
            },
        'style_vec': {
            'repo_id':'litagin/style_bert_vits2_jvnv',
            'path': 'jvnv-M2-jp/style_vectors.npy'
            },
        'language': 'jp',
        'model_name': 'JVNV M2',
        'description': 'JVNVコーパス男性話者2',
        'gender': 'male',
        'license': 'CC BY-SA 4.0（JVNVコーパス継承）',
        'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/deed.ja',
        'usage_terms': 'JVNVコーパス由来のためCC BY-SA 4.0を継承。表示・継承条件を満たして利用してください。',
    },
    'rinne': {
        'model':{
            'repo_id':'RinneAi/Rinne_Style-Bert-VITS2',
            'path':'model_assets/Rinne/Rinne.safetensors'
            },
        'config': {
            'repo_id':'RinneAi/Rinne_Style-Bert-VITS2',
            'path': 'model_assets/Rinne/config.json'
            },
        'style_vec': {
            'repo_id':'RinneAi/Rinne_Style-Bert-VITS2',
            'path': 'model_assets/Rinne/style_vectors.npy'
            },
        'language': 'jp',
        'model_name': 'りんねおねんね',
        'gender': 'female',
        'license': '配布者記載: 商用・非商用問わず利用可',
        'license_url': 'https://booth.pm/ja/items/6919603?srsltid=AfmBOooFYrF78FW-NrbuG0UZWVenOcs8010gOECKnHCUFNpjxlzmfbyC',
        'usage_terms': '配布ページ記載に基づき、商用・非商用問わず利用可能。詳細条件・最新情報は配布ページの記載を確認してください。',
    },
    'NotAnimeJPManySpeaker': {
        'model':{
            'repo_id': 'Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
            'path': 'NotAnimeJPManySpeaker_e120_s22200.safetensors'
            },
        'config': {
            'repo_id': 'Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
            'path': 'config.json'
            },
        'style_vec': {
            'repo_id': 'Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
            'path': 'style_vectors.npy'
            },
        'language': 'jp',
        'model_name': 'girl（JP-Extra 多話者）',
        'license': '要確認（配布ページ参照）',
        'license_url': 'https://huggingface.co/Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
        'usage_terms': '利用前に配布ページの利用規約とライセンスを確認し、禁止事項（再配布、商用利用、二次配布、用途制限など）を遵守してください。',
    },
    'tsukuyomi-chan': {
        'model':{
            'repo_id': 'ayousanz/tsukuyomi-chan-style-bert-vits2-model',
            'path': 'tsukuyomi-chan_e116_s3000.safetensors'
            },
        'config': {
            'repo_id': 'ayousanz/tsukuyomi-chan-style-bert-vits2-model',
            'path': 'config.json'
            },
        'style_vec': {
            'repo_id': 'ayousanz/tsukuyomi-chan-style-bert-vits2-model',
            'path': 'style_vectors.npy'
            },
        'language': 'jp',
        'model_name': 'つくよみちゃん',
        'gender': 'female',
        'license': 'つくよみちゃん利用規約（公式サイト参照）',
        'license_url': 'https://tyc.rei-yumesaki.net/about/terms/',
        'usage_terms': '利用時は公式の「つくよみちゃん利用規約」に従ってください。商用利用・再配布・クレジット要否などの詳細条件は必ず公式規約本文を確認してください。',
    },
    'AbeShinzo': {
        'model':{
            'repo_id': 'AbeShinzo0708/AbeShinzo_Style_Bert_VITS2',
            'path': 'AbeShinzo20240210_e300_s43800.safetensors'
            },
        'config': {
            'repo_id': 'AbeShinzo0708/AbeShinzo_Style_Bert_VITS2',
            'path': 'config.json'
            },
        'style_vec': {
            'repo_id': 'AbeShinzo0708/AbeShinzo_Style_Bert_VITS2',
            'path': 'style_vectors.npy'
            },
        'language': 'jp',
        'model_name': '安倍晋三',
        'description': '安倍晋三元首相の音声データを用いたモデル',
        'gender': 'male',
        'license': 'Apache License 2.0',
        'license_url': 'https://www.apache.org/licenses/LICENSE-2.0',
        'usage_terms': '安倍晋三元首相の音声データを用いたモデルです。フェイクニュース・誹謗中傷・名誉毀損につながる利用、誤解を招くコンテンツ作成は禁止。公序良俗に反する用途や権利侵害の恐れがある利用は避けてください。',
    },
    'sakura-miko': {
        'model':{
            'repo_id': 'Lycoris53/style-bert-vits2-sakura-miko',
            'path': 'sakuramiko_e89_s23000.safetensors'
            },
        'config': {
            'repo_id': 'Lycoris53/style-bert-vits2-sakura-miko',
            'path': 'config.json'
            },
        'style_vec': {
            'repo_id': 'Lycoris53/style-bert-vits2-sakura-miko',
            'path': 'style_vectors.npy'
            },
        'language': 'jp',
        'model_name': 'さくらみこ',
        'gender': 'female',
        'license': '配布者条件: 趣味の範囲で利用',
        'license_url': 'https://hololivepro.com/terms/',
        'usage_terms': 'モデルの取得や使い方は自由ですが、趣味の範囲で利用してください。詳細はカバー株式会社の二次創作ガイドライン（https://hololivepro.com/terms/）を確認してください。',
    }
}

def download_sbv2_model(arg) -> Path:
    if isinstance(arg, dict):
        return Path(download_hf_hub(**arg))
    else:
        return Path(arg)

def to_language(lang: str|None) -> Languages:
    if lang and "en" in lang.lower():
        return Languages.EN
    elif lang and "zh" in lang.lower():
        return Languages.ZH
    else:
        return Languages.JP

@dataclass
class SpkOptions:
    model: int | str | None = None
    model_path: str|Path|None = None
    config_path: str|Path|None = None
    style_vec_path: str|Path|None = None
    model_repo: str|None = None
    speaker_id: int | None = None
    speaker_style: str | None = None
    speaker_name: str | None = None
    split: bool = False
    speedScale: float = 1.0
    pitchOffset: float = 0.0
    lang: str = "ja"
    model_name: str|None = None
    gender: str = "unknown"
    description: str=''
    license_name: str|None = None
    license_url: str|None = None
    usage_terms: str|None = None

@lru_cache(maxsize=1)
def get_sbv2_options_list() -> dict[str,SpkOptions]:
    def _spk_option(model_key: str, *, speaker_id: int|None=None, model_name:str|None=None, gender:str|None=None, description:str|None=None) -> SpkOptions:
        assert isinstance(model_key, str), f"model_key must be str, got {type(model_key)}"
        model_info = SBV2_MODELS.get(model_key, {})
        cfg_model_name = model_name or model_info.get("model_name") or model_key
        cfg_gender = gender if gender in ('male', 'female') else model_info.get('gender') or 'unknown'
        cfg_description = description if description is not None else model_info.get('description') or ''
        model_repo = model_info.get("model", {}).get("repo_id")
        model_path = model_info.get("model", {}).get("path")
        config_path = model_info.get("config", {}).get("path")
        style_vec_path = model_info.get("style_vec", {}).get("path")
        lang = model_info.get("language") or "unknown"
        return SpkOptions(
            model=model_key,
            model_repo=model_repo,
            model_path=model_path,
            config_path=config_path,
            style_vec_path=style_vec_path,
            speaker_id=speaker_id,
            model_name=cfg_model_name,
            gender=cfg_gender,
            license_name=model_info.get("license"),
            license_url=model_info.get("license_url"),
            usage_terms=model_info.get("usage_terms"),
            description=cfg_description,
            lang=lang,
        )
    XX1 = 'アニメ調じゃない日本語多話者モデル'
    return {
        "amitaro": _spk_option("amitaro", speaker_id=0),
        "koharune-ami": _spk_option("koharune-ami", speaker_id=0),
        "amazinGood": _spk_option("NotAnimeJPManySpeaker", speaker_id=0, model_name="amazinGood",gender="female",description=XX1),
        "calmCloud": _spk_option("girNotAnimeJPManySpeakerl", speaker_id=1, model_name="calmCloud",gender="female",description=XX1),
        "coolcute": _spk_option("NotAnimeJPManySpeaker", speaker_id=2, model_name="coolcute",gender="female",description=XX1),
        "fineCrystal": _spk_option("NotAnimeJPManySpeaker", speaker_id=3, model_name="fineCrystal",gender="female",description=XX1),
        "lightFire": _spk_option("NotAnimeJPManySpeaker", speaker_id=4, model_name="lightFire",gender="male",description=XX1),
        "Rinne": _spk_option("rinne"),
        "AbeShinzo": _spk_option("AbeShinzo"),
        "tsukuyomi-chan": _spk_option("tsukuyomi-chan"),
        "sakura-miko": _spk_option("sakura-miko"),
        "jvnv-jp-F1": _spk_option("jvnv-F1-jp"),
        "jvnv-jp-F2": _spk_option("jvnv-F2-jp"),
        "jvnv-jp-M1": _spk_option("jvnv-M1-jp"),
        "jvnv-jp-M2": _spk_option("jvnv-M2-jp"),
    }


def print_help():
    pass

TEXT_PREPROCESS_RECOMMENDATION = (
    "【読み上げ前のテキスト前処理の推奨】\n"
    "- アルファベット表記は、できるだけカタカナに変換してから入力してください。\n"
    "- 読み間違いやすい漢字（例: 「方」「日」など）は、ひらがなに変換してから入力するほうが望ましいです。"
)

def print_text_preprocess_recommendation() -> None:
    print(TEXT_PREPROCESS_RECOMMENDATION)

def print_model_detail(model_name: str, model_info: dict) -> None:
    print(f"[{model_name}]")
    print(f"  モデル名: {model_info.get('model_name', model_name)}")
    print(f"  言語: {model_info.get('language', 'unknown')}")
    print(f"  モデル配布: https://huggingface.co/{model_info.get('model', {}).get('repo_id', 'unknown')}")
    print(f"  ライセンス表記: {model_info.get('license', '未設定')}")
    print(f"  ライセンスURL: {model_info.get('license_url', '未設定')}")
    print(f"  利用条件: {model_info.get('usage_terms', '未設定')}")

def print_models_doc() -> int:
    options = get_sbv2_options_list()
    print("# --modelオプションで指定できるモデルの一覧です。model-idを指定して下さい。")
    print("|model-id|model_name|gender|lang|description|")
    print("|---|---|---|---|---|")
    for key, opt in options.items():
        print(f"|{key}|{opt.model_name}|{opt.gender}|{opt.lang}|{opt.description or ''}|")
    print("")
    return 0

def print_model_info_doc(model_id: str) -> int:
    options = get_sbv2_options_list()
    option = options.get(model_id)
    if option is None:
        print(f"モデルが見つかりません: {model_id}", file=sys.stderr)
        return 1

    model_repo = option.model_repo or "unknown"
    model_path = option.model_path or "unknown"
    config_path = option.config_path or "unknown"
    style_vec_path = option.style_vec_path or "unknown"
    language = option.lang or "unknown"

    output_lines = [
        f"# {model_id}",
        "",
        f"- model_name: {option.model_name or model_id}",
        f"- language: {language}",
        f"- model_repo: https://huggingface.co/{model_repo}",
        f"- safetensors: {model_path}",
        f"- config: {config_path}",
        f"- style_vectors: {style_vec_path}",
        f"- license: {option.license_name or '未設定'}",
        f"- license_url: {option.license_url or '未設定'}",
        f"- usage_terms: {option.usage_terms or '未設定'}",
        "",
    ]
    print("\n".join(output_lines).rstrip())
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Style-Bert-VITS2 CLI",
        epilog=TEXT_PREPROCESS_RECOMMENDATION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-?", action="help", help="show this help message and exit")
    parser.add_argument("--text", action="append", help="input text (repeatable)")
    parser.add_argument("--model", default="amitaro", help="model name")
    parser.add_argument("--output", action="append", help="output wav path (repeatable)")
    parser.add_argument("--list-models", "--models", action="store_true", dest="list_models", help="list available models")
    parser.add_argument("--model-info", metavar="MODEL_ID", help="show model info from docs")
    parser.add_argument("--spkeaker-id", type=int, default=None, help="speaker ID")
    parser.add_argument("--speed", type=float, help="speed scale")
    parser.add_argument("--sr", type=int, default=24000, help="output sample rate")
    args = parser.parse_args()

    if args.model_info:
        return print_model_info_doc(args.model_info)

    if args.list_models:
        return print_models_doc()

    texts = args.text or ["こんにちは"]
    if args.output is None:
        if len(texts) == 1:
            outputs = ["output.wav"]
        else:
            outputs = [f"output_{i + 1:03d}.wav" for i in range(len(texts))]
    else:
        outputs = args.output
    if len(texts) != len(outputs):
        parser.error(
            f"--text の数 ({len(texts)}) と --output の数 ({len(outputs)}) を一致させてください。"
        )

    model_dict = get_sbv2_options_list()
    mdl = model_dict.get(args.model)
    if not args.model:
        model_dict = SBV2_MODELS["amitaro"]
    else:
        model_dict = SBV2_MODELS.get(args.model)
    if model_dict is None:
        print(f"Model {args.model} not found. Using")
        return 1

    print_model_detail(args.model, model_dict)
    print(f"num_jobs={len(texts)}")
    print(f"sample rate={args.sr}")
    print_text_preprocess_recommendation()

    model_path = model_dict.get('model')
    config_path = model_dict.get('config')
    style_vec_path = model_dict.get('style_vec')
    language = to_language(model_dict.get('language'))

    bert_models.load_model(language, SBV2_TOKENIZER_PATHS[language])
    bert_models.load_tokenizer(language, SBV2_TOKENIZER_PATHS[language])

    a_model_path = download_sbv2_model(model_path)
    a_config_path = download_sbv2_model(config_path)
    a_style_vec_path = download_sbv2_model(style_vec_path)
    #
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tts_model = SBV2_TTSModel( device=device,
        model_path= a_model_path,
        config_path= a_config_path,
        style_vec_path= a_style_vec_path,
    )
    tts_model.load()

    # if len(tts_model.id2spk)>1:
    # print("speakers information :")
    # for k,v in tts_model.id2spk.items():
    #     print(f"id:{k} name:{v}")
    # for k,v in tts_model.style2id.items():
    #     print(f"style_name={k} style_id={v}")
    # print(f"styles: {','.join([k for k in tts_model.style2id.keys()])}")

    speaker_id:int = 0 if 0 in tts_model.id2spk else list(tts_model.id2spk.keys())[0]
    speaker_name:str = tts_model.id2spk[speaker_id]
    speaker_style:str = DEFAULT_STYLE if DEFAULT_STYLE in tts_model.style2id else list(tts_model.style2id.keys())[0]
    speedScale:float = 1.0

    # if args.speaker_id in tts_model.id2spk:
    #     speaker_id = args.speaker_id
    #     speaker_name = tts_model.id2spk[speaker_id]
    # elif args.speaker_name in tts_model.spk2id:
    #     speaker_id = tts_model.spk2id[args.speaker_name]
    #     speaker_name = args.speaker_name

    # if args.speaker_style in tts_model.style2id:
    #     speaker_style = args.speaker_style

    # if args.speed:
    #     speedScale = speedScale * args.speed

    for idx, (text, output_path) in enumerate(zip(texts, outputs), start=1):
        print(f"[{idx}/{len(texts)}] output={output_path}")
        tts_sr, audio_i16 = tts_model.infer(
            text,
            speaker_id=speaker_id,style=speaker_style,
            # assist_text=self._assist_text,use_assist_text=True
        )
        if tts_sr != args.sr:
            # print(f"Warning: sample rate mismatch {tts_sr} != {args.sr}")
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
            resampled_f32 = librosa.resample(audio_f32, orig_sr=tts_sr, target_sr=args.sr)
            audio_i16 = (resampled_f32 * 32768.0).clip(min=-32768, max=32767).astype(np.int16)

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path_obj), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(args.sr)
            wf.writeframes(audio_i16.tobytes())
        print(f"  done: {output_path_obj}")

if __name__ == "__main__":
    main()
