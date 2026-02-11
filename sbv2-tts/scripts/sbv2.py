
import sys,os
import re
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field
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

from style_bert_vits2.logging import logger as sbv2_logger
sbv2_logger.remove()  # 既存のログ設定を削除
sbv2_logger.add(sys.stderr, level="ERROR")  # ERRORレベルのログのみを表示

from style_bert_vits2.constants import DEFAULT_BERT_TOKENIZER_PATHS, Languages, DEFAULT_STYLE, DEFAULT_LENGTH
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel as SBV2_TTSModel

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

import alkana

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

@dataclass
class RepoFile:
    repo_id: str
    path: str

    def download(self) -> Path:
        return Path(download_hf_hub(self.repo_id, self.path))

@dataclass
class ModelInfo:
    id: str
    name: str
    spker_id: int = 0
    gender: str = "unknown"
    language: Languages = Languages.JP
    description: str = ""
    styles: dict[str, int] = field(default_factory=lambda: {DEFAULT_STYLE: 0})
    speedScale: float = 1.0
    pitchOffset: float = 0.0

@dataclass
class DataSet:
    safetensors: RepoFile
    config: RepoFile
    style_vectors: RepoFile
    models: list[ModelInfo]
    license: str = "unknown"
    license_url: str = "unknown"
    usage_terms: str = "unknown"

@lru_cache(maxsize=1)
def get_datasets() -> list[DataSet]:
    return [
        DataSet(
            safetensors=RepoFile(
                repo_id='litagin/sbv2_amitaro',
                path='amitaro/amitaro.safetensors'
            ),
            config=RepoFile(
                repo_id='litagin/sbv2_amitaro',
                path='amitaro/config.json'
            ),
            style_vectors=RepoFile(
                repo_id='litagin/sbv2_amitaro',
                path='amitaro/style_vectors.npy'
            ),
            models=[ModelInfo(id='amitaro', name='あみたろ', gender='female',description='配信向きかわいい声',)],
            license='あみたろの声素材工房規約（配布元規約準拠）',
            license_url='https://amitaro.net/voice/voice_rule/',
            usage_terms='amitaro.netの規約（voice_rule と livevoice）を遵守。年齢制限用途・政治/宗教/マルチ・誹謗中傷用途は禁止。公開時は「あみたろの声素材工房 (https://amitaro.net/)」のクレジット表記が必要。'
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id='litagin/sbv2_koharune_ami',
                path='koharune-ami/koharune-ami.safetensors'
            ),
            config=RepoFile(
                repo_id='litagin/sbv2_koharune_ami',
                path='koharune-ami/config.json'
            ),
            style_vectors=RepoFile(
                repo_id='litagin/sbv2_koharune_ami',
                path='koharune-ami/style_vectors.npy'
            ),
            models=[ModelInfo(id='koharune-ami', name='小春音アミ', gender='female',description='配信向きかわいい声',
                styles={DEFAULT_STYLE:0, 'るんるん':1, 'ささやきA(無声)': 2, 'ささやきB(有声)': 3, 'ノーマル':4, 'よふかし':5})],
            license='あみたろの声素材工房規約（配布元規約準拠）',
            license_url='https://amitaro.net/voice/voice_rule/',
            usage_terms='amitaro.netの規約（voice_rule と livevoice）を遵守。年齢制限用途・政治/宗教/マルチ・誹謗中傷用途は禁止。公開時は「あみたろの声素材工房 (https://amitaro.net/)」のクレジット表記が必要。'
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors'
            ),
            config=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-F1-jp/config.json'
            ),
            style_vectors=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-F1-jp/style_vectors.npy'
            ),
            models=[ModelInfo(id='jvnv-F1-jp', name='JVNV F1', gender='female', description='JVNVコーパス女性話者1',
                              styles={DEFAULT_STYLE:0, 'Angry':1, 'Disgust':2, 'Fear':3, 'Happy':4, 'Sad':5, 'suprise':6})],
            license='CC BY-SA 4.0（JVNVコーパス継承）',
            license_url='https://creativecommons.org/licenses/by-sa/4.0/deed.ja',
            usage_terms='JVNVコーパス由来のためCC BY-SA 4.0を継承。表示・継承条件を満たして利用してください。',
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-F2-jp/jvnv-F2_e166_s20000.safetensors'
            ),
            config=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-F2-jp/config.json'
            ),
            style_vectors=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-F2-jp/style_vectors.npy'
            ),
            models=[ModelInfo(id='jvnv-F2-jp', name='JVNV F2', gender='female', description='JVNVコーパス女性話者2',
                              styles={DEFAULT_STYLE:0, 'Angry':1, 'Disgust':2, 'Fear':3, 'Happy':4, 'Sad':5, 'suprise':6})],
            license='CC BY-SA 4.0（JVNVコーパス継承）',
            license_url='https://creativecommons.org/licenses/by-sa/4.0/deed.ja',
            usage_terms='JVNVコーパス由来のためCC BY-SA 4.0を継承。表示・継承条件を満たして利用してください。',
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors'
            ),
            config=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-M1-jp/config.json'
            ),
            style_vectors=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-M1-jp/style_vectors.npy'
            ),
            models=[ModelInfo(id='jvnv-M1-jp', name='JVNV M1', gender='male', description='JVNVコーパス男性話者1',
                              styles={DEFAULT_STYLE:0, 'Angry':1, 'Disgust':2, 'Fear':3, 'Happy':4, 'Sad':5, 'suprise':6})],
            license='CC BY-SA 4.0（JVNVコーパス継承）',
            license_url='https://creativecommons.org/licenses/by-sa/4.0/deed.ja',
            usage_terms='JVNVコーパス由来のためCC BY-SA 4.0を継承。表示・継承条件を満たして利用してください。',
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-M2-jp/jvnv-M2-jp_e159_s17000.safetensors'
            ),
            config=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-M2-jp/config.json'
            ),
            style_vectors=RepoFile(
                repo_id='litagin/style_bert_vits2_jvnv',
                path='jvnv-M2-jp/style_vectors.npy'
            ),
            models=[ModelInfo(id='jvnv-M2-jp', name='JVNV M2', gender='male', description='JVNVコーパス男性話者2',
                              styles={DEFAULT_STYLE:0, 'Angry':1, 'Disgust':2, 'Fear':3, 'Happy':4, 'Sad':5, 'suprise':6})],
            license='CC BY-SA 4.0（JVNVコーパス継承）',
            license_url='https://creativecommons.org/licenses/by-sa/4.0/deed.ja',
            usage_terms='JVNVコーパス由来のためCC BY-SA 4.0を継承。表示・継承条件を満たして利用してください。',
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id='RinneAi/Rinne_Style-Bert-VITS2',
                path='model_assets/Rinne/Rinne.safetensors'
            ),
            config=RepoFile(
                repo_id='RinneAi/Rinne_Style-Bert-VITS2',
                path='model_assets/Rinne/config.json'
            ),
            style_vectors=RepoFile(
                repo_id='RinneAi/Rinne_Style-Bert-VITS2',
                path='model_assets/Rinne/style_vectors.npy'
            ),
            models=[ModelInfo(id='rinne', name='りんねおねんね', gender='female', description='すこし幼い可愛い声')],
            license='配布者記載: 商用・非商用問わず利用可',
            license_url='https://booth.pm/ja/items/6919603?srsltid=AfmBOooFYrF78FW-NrbuG0UZWVenOcs8010gOECKnHCUFNpjxlzmfbyC',
            usage_terms='配布ページ記載に基づき、商用・非商用問わず利用可能。詳細条件・最新情報は配布ページの記載を確認してください。',
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id='Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
                path='NotAnimeJPManySpeaker_e120_s22200.safetensors'
            ),
            config=RepoFile(
                repo_id='Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
                path='config.json'
            ),
            style_vectors=RepoFile(
                repo_id='Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
                path='style_vectors.npy'
            ),
            models=[
                ModelInfo(id='amazinGood', name='amazinGood', spker_id=0, gender='female',description='20代女性の声、感情抑えめルーズな話し方',
                          styles={DEFAULT_STYLE:4, 'down':1, 'lol':2, 'ohmygod':3}),
                ModelInfo(id='calmCloud', name='calmCloud', spker_id=1, gender='female',description='20代女性の声',
                          styles={DEFAULT_STYLE:10, 'lol':5, 'question':6, 'down':7, 'hate': 8, 'ohmygod':9}),
                ModelInfo(id='coolcute', name='coolcute', spker_id=2, gender='female',description='20代女性の声',
                          styles={DEFAULT_STYLE:12, 'ohmygod':11, 'fine':13, 'sad':14}),
                ModelInfo(id='fineCrystal', name='fineCrystal', spker_id=3, gender='female',description='20代女性の声',
                          styles={DEFAULT_STYLE:18, 'fine':15, 'ohmygod':16, 'veryfine':17, 'sad':19}),
                ModelInfo(id='lightFire', name='lightFire', spker_id=4, gender='male',description='20代男性の声',
                          styles={DEFAULT_STYLE:22, 'question':20, 'hello':21, 'strong':23, 'lol':24}),
            ],
            license='要確認（配布ページ参照）',
            license_url='https://huggingface.co/Mofa-Xingche/girl-style-bert-vits2-JPExtra-models',
            usage_terms='利用前に配布ページの利用規約とライセンスを確認し、禁止事項（再配布、商用利用、二次配布、用途制限など）を遵守してください。',
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id= 'ayousanz/tsukuyomi-chan-style-bert-vits2-model',
                path='tsukuyomi-chan_e116_s3000.safetensors'
            ),
            config=RepoFile(
                repo_id= 'ayousanz/tsukuyomi-chan-style-bert-vits2-model',
                path='config.json'
            ),
            style_vectors=RepoFile(
                repo_id= 'ayousanz/tsukuyomi-chan-style-bert-vits2-model',
                path='style_vectors.npy'
            ),
            models=[ModelInfo(id='tsukuyomi-chan', name='つくよみちゃん', gender='female')],
            license='つくよみちゃん利用規約（公式サイト参照）',
            license_url='https://tyc.rei-yumesaki.net/about/terms/',
            usage_terms='利用時は公式の「つくよみちゃん利用規約」に従ってください。商用利用・再配布・クレジット要否などの詳細条件は必ず公式規約本文を確認してください。',
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id='AbeShinzo0708/AbeShinzo_Style_Bert_VITS2',
                path='AbeShinzo20240210_e300_s43800.safetensors'
            ),
            config=RepoFile(
                repo_id='AbeShinzo0708/AbeShinzo_Style_Bert_VITS2',
                path='config.json'
            ),
            style_vectors=RepoFile(
                repo_id='AbeShinzo0708/AbeShinzo_Style_Bert_VITS2',
                path='style_vectors.npy'
            ),
            models=[ModelInfo(id='AbeShinzo', name='安倍晋三', gender='male',description='安倍晋三元首相の音声データを用いたモデル',
                              styles={DEFAULT_STYLE:0, 'Angry':1, 'Sad':2, 'Noisy':3, 'Clam': 4})],
            license='Apache License 2.0',
            license_url='https://www.apache.org/licenses/LICENSE-2.0',
            usage_terms='安倍晋三元首相の音声データを用いたモデルです。フェイクニュース・誹謗中傷・名誉毀損につながる利用、誤解を招くコンテンツ作成は禁止。公序良俗に反する用途や権利侵害の恐れがある利用は避けてください。',
        ),
        DataSet(
            safetensors=RepoFile(
                repo_id='Lycoris53/style-bert-vits2-sakura-miko',
                path='sakuramiko_e89_s23000.safetensors'
            ),
            config=RepoFile(
                repo_id='Lycoris53/style-bert-vits2-sakura-miko',
                path='config.json'
            ),
            style_vectors=RepoFile(
                repo_id='Lycoris53/style-bert-vits2-sakura-miko',
                path='style_vectors.npy'
            ),
            models=[ModelInfo(id='sakura-miko', name='さくらみこ', gender='female',
                              styles={DEFAULT_STYLE:0, 'Happy':1, 'Sad':2, 'Angry':3})],
            license='配布者条件: 趣味の範囲で利用',
            license_url='https://hololivepro.com/terms/',
            usage_terms='モデルの取得や使い方は自由ですが、趣味の範囲で利用してください。詳細はカバー株式会社の二次創作ガイドライン（https://hololivepro.com/terms/）を確認してください。',
        ),
    ]

TEXT_PREPROCESS_RECOMMENDATION = (
    "【読み上げ前のテキスト前処理の推奨】\n"
    "- アルファベット表記は、できるだけカタカナに変換してから入力してください。\n"
    "- 読み間違いやすい漢字（例: 「方」「日」など）は、ひらがなに変換してから入力するほうが望ましいです。\n"
    "# クレジット表記を推奨\n"
    "- 資料や動画には出来るだけ音声のクレジット表示をお願いします。\n"
    "   例: CV: モデル名(https://......)"
)
def get_model_by_id(model_id: str) -> tuple[DataSet|None, ModelInfo|None]:
    data_list = get_datasets()
    for ds in data_list:
        for mdl in ds.models:
            if mdl.id == model_id:
                return ds, mdl
    return None, None

def get_default_model() -> tuple[DataSet, ModelInfo]:
    datasets = get_datasets()
    ds = datasets[0]
    mdl = ds.models[0]
    return ds, mdl

def load_model(dataset: DataSet, device: str|None = None) -> SBV2_TTSModel:

    a_model_path = dataset.safetensors.download()
    a_config_path = dataset.config.download()
    a_style_vec_path = dataset.style_vectors.download()

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tts_model = SBV2_TTSModel(
        device=device,
        model_path=a_model_path,
        config_path=a_config_path,
        style_vec_path=a_style_vec_path,
    )
    tts_model.load()
    return tts_model

def print_text_preprocess_recommendation() -> None:
    print(TEXT_PREPROCESS_RECOMMENDATION)

def print_model_list() -> int:
    data_list = get_datasets()
    print("# --modelオプションで指定できるモデルの一覧です。model-idを指定して下さい。")
    print("|model-id|model_name|gender|lang|description|")
    print("|---|---|---|---|---|")
    for ds in data_list:
        for model in ds.models:
            print(f"|{model.id}|{model.name}|{model.gender}|{model.language.name}|{model.description or ''}|")
    print("")
    print(TEXT_PREPROCESS_RECOMMENDATION)
    print("")
    return 0

def print_model_info( dataset: DataSet, model: ModelInfo) -> None:
    output_lines = [
        f"- model_id: {model.id}",
        f"- model_name: {model.name} gender: {model.gender} lang: {model.language.name}",
        f"- description: {model.description or '未設定'}",
        f"- styles: {', '.join([k for k in model.styles.keys()])}",
        f"- license: {dataset.license or '未設定'}",
        f"- license_url: {dataset.license_url or '未設定'}",
        f"- usage_terms: {dataset.usage_terms or '未設定'}",
        "",
    ]
    print("\n".join(output_lines).rstrip())
    print("")
    print(TEXT_PREPROCESS_RECOMMENDATION)
    print("")

def verify_models():
    device = "cpu"
    data_list = get_datasets()
    for dataset in data_list:
        print("========================================")
        print(f"dataset model(s): {[model.id for model in dataset.models]}")
        tts_model = load_model(dataset, device=device)
        print("    speakers information :")
        for style_name,style_id in tts_model.id2spk.items():
            print(f"        id:{style_name} name:{style_id}")
        print("    styles information :")
        for style_name,style_id in tts_model.style2id.items():
            print(f"        style_name={style_name} style_id={style_id}")

        for model in dataset.models:
            print(f"    model id:{model.id} name:{model.name} spker_id:{model.spker_id}")
            if model.spker_id in tts_model.id2spk:
                speaker_id = model.spker_id
                speaker_name = tts_model.id2spk[speaker_id]
                print(f"        speaker_id:{speaker_id} speaker_name:{speaker_name}")
            else:
                print(f"        ERROR: speaker_id:{model.spker_id} not found in tts_model")
            if DEFAULT_STYLE not in model.styles:
                print(f"        ERROR: default style '{DEFAULT_STYLE}' not found in model.styles")
            for style_name, style_id in model.styles.items():
                a = None
                for k,v in tts_model.style2id.items():
                    if v == style_id:
                        a = k
                if a:
                    print(f"        style_name:{style_name} style_id:{style_id} => {a}")
                else:
                    print(f"        ERROR: style_name:{style_name} style_id:{style_id} not found in tts_model")

def to_halfwidth(s: str) -> str:
    trans = str.maketrans(
        "０１２３４５６７８９"
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
        "，．！？＃＄％＆ー＝＾〜￥｜＠：＊／＿；＋＜＞（）［］｛｝”“‘’｀",
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        ",.!?#$%&-=^~¥|@:*/_;+<>()[]{}\"\"''`",
    )
    return s.translate(trans)

def to_kana(word: str) -> str:
    k = alkana.get_kana(word)
    return k if k else word

def english_to_katakana(text: str) -> str:
    # 簡易的な英語→カタカナ変換
    # textから正規表現で英単語を抽出し、辞書で変換
    def replace_match(match):
        word = match.group(0).lower()
        katakana = ""
        for w in word.replace("-", "_").split("_"):
            k = to_kana(w)
            katakana += k if k else w
        return katakana

    pattern = re.compile(r'\b[a-z_-]+\b', re.IGNORECASE)
    return pattern.sub(replace_match, text)

def split_to_chunks(sentence: str) -> list[tuple[str,bool]]:
    # 改行、句点で分割
    segments = []
    slen = len(sentence)
    start = 0
    for p,c in enumerate(sentence):
        if p+1==slen or c==sentence[p+1]:
            continue
        is_split = 0
        if c in ("\n","。","、"):
            is_split = 2
        elif c in ("、"):
            is_split = 1
        elif c in (".","!","?") and sentence[p+1] == " ":
            is_split = 2
        if is_split>0:
            segment = sentence[start:p+1].strip()
            if segment:
                segments.append((segment, is_split==2))
            start = p+1
    if start < len(sentence):
        segment = sentence[start:].strip()
        if segment:
            segments.append((segment, True))
    return segments

def split_to_sentences( text: str, width:int = 90 ) -> list[str]:
    results = []
    buffer = ""
    for segment,is_split in split_to_chunks(text):
        sumlen = len(buffer) + len(segment)
        if sumlen > width:
            if buffer:
                results.append(buffer.strip())
            buffer = segment + " "
        buffer += segment
        if is_split:
            if buffer:
                results.append(buffer.strip())
            buffer = ""
    if buffer:
        results.append(buffer.strip())
    return results

def apply_filter( audio: NDArray[np.int16], sr:int) -> NDArray[np.int16]:
    # 簡易的なノイズ除去フィルター
    if audio.size == 0:
        return audio

    # バンドパスフィルタを適用（人声向け: 120Hz～4000Hz）
    audio_f32 = audio.astype(np.float32) / 32768.0
    stft_matrix = librosa.stft(audio_f32)

    # STFTの0軸は周波数ビン
    n_fft = (stft_matrix.shape[0] - 1) * 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 周波数以外の成分をゼロ化
    mask = (freqs >= 120) & (freqs <= 4000)
    stft_matrix[~mask, :] = 0

    # 元の長さに合わせて逆変換
    audio_filtered = librosa.istft(stft_matrix, length=audio_f32.shape[0])
    return (audio_filtered * 32768.0).clip(min=-32768, max=32767).astype(np.int16)

def main():
    parser = argparse.ArgumentParser(
        description="Style-Bert-VITS2 CLI",
        epilog=TEXT_PREPROCESS_RECOMMENDATION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-?", action="help", help="show this help message and exit")
    parser.add_argument("--model", default="amitaro", help="model name")
    parser.add_argument("--style", type=str, default=None, help=f"speaker style(default: {DEFAULT_STYLE})")
    parser.add_argument("--speed", type=float, help="speed scale(default: 1.0, 0.5-2.0 recommended)")
    parser.add_argument("--text", action="append", help="input text (repeatable)")
    parser.add_argument("--assist-text", type=str, default=None,help="Auxiliary text that only affects BERT embeddings. Should be short.Affects speaking tone, does not affect text analysis.")
    parser.add_argument("--output", action="append", help="output wav path (repeatable)")
    parser.add_argument("--sr", type=int, default=24000, help="output sample rate(default: 24000)")
    parser.add_argument("--list-models", action="store_true", dest="list_models", help="list available models")
    parser.add_argument("--model-info", metavar="MODEL_ID", help="show model info and styles")
    parser.add_argument("--verify-models", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("files", nargs="*", metavar="TXT", help="input text file(s) (.txt)")
    args = parser.parse_args()

    # --list-models オプション
    if args.list_models:
        return print_model_list()

    # --model-info オプション
    if args.model_info:
        dataset, model = get_model_by_id(args.model_info)
        if dataset is None or model is None:
            print(f"モデルが見つかりません: {args.model_info}", file=sys.stderr)
            return 1
        print_model_info(dataset, model)
        return 0

    # --verify-models オプション
    if args.verify_models:
        verify_models()
        return 0

    # 入力テキストと出力ファイルの設定
    if args.files:
        if args.text:
            parser.error("--text と TXT ファイルは同時に指定できません。")
        if args.output:
            parser.error("TXT ファイル指定時は --output を使えません。")
        texts = []
        outputs = []
        for file_path in args.files:
            path = Path(file_path)
            if path.suffix.lower() != ".txt":
                parser.error(f"TXT ファイル以外は指定できません: {file_path}")
            try:
                text = path.read_text(encoding="utf-8").strip()
                text = to_halfwidth(text)
            except OSError as exc:
                print(f"TXT ファイルを読み込めません: {file_path} ({exc})", file=sys.stderr)
                return 1
            texts.append(text)
            outputs.append(str(path.with_suffix(".wav")))
    else:
        texts = [to_halfwidth(t) for t in (args.text or ["こんにちは"])]
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

    if not args.model:
        dataset, model = get_default_model()
    else:
        dataset, model = get_model_by_id(args.model)

    if dataset is None or model is None:
        print(f"Model {args.model} not found. Using")
        return 1

    style_name = args.style or DEFAULT_STYLE
    if style_name not in model.styles:
        print(f"Style '{style_name}' not found in model '{model.id}'.")
        print(f"Available styles: {', '.join(model.styles.keys())}")
        return 1

    print_model_info( dataset, model)

    language = model.language
    bert_models.load_model(language, SBV2_TOKENIZER_PATHS[language])
    bert_models.load_tokenizer(language, SBV2_TOKENIZER_PATHS[language])

    device: str|None = None
    tts_model = load_model(dataset, device=device)

    speaker_id:int = model.spker_id
    speaker_style:str = style_name

    if model.speedScale <= 0:
        print(f"Model speedScale must be > 0: {model.speedScale}", file=sys.stderr)
        return 1

    if args.speed is not None:
        if args.speed <= 0:
            parser.error("--speed は 0 より大きい値を指定してください。")
        length_ratio = round(model.speedScale / args.speed, 6)
    else:
        length_ratio = round(model.speedScale, 6)

    length_scale = max(0.5, min(2.0, round( DEFAULT_LENGTH * length_ratio, 2)))
    if 0.99 < length_scale < 1.01:
        length_scale = None  # デフォルト値に近い場合は None にする

    pitch_scale = 1.0 + model.pitchOffset
    if 0.99 < pitch_scale < 1.01:
        pitch_scale = None  # デフォルト値に近い場合は None にする

    assist_text: str|None = None
    if isinstance(args.assist_text, str) and len(args.assist_text.strip())>0:
        assist_text = args.assist_text.strip()

    text_max_width = 90
    for idx, (text, output_path) in enumerate(zip(texts, outputs), start=1):
        print(f"[{idx}/{len(texts)}] output={output_path}")
        # テキストを分割
        segments = split_to_sentences(text, width=text_max_width)

        if not segments:
            print(f"入力テキストが空です: output={output_path}", file=sys.stderr)
            return 1

        total_audio_i16 = []
        total_length:int = 0
        timings: list[tuple[int,int,str]] = []
        for i,seg in enumerate(segments):
            kseg = english_to_katakana(seg)
            if seg!= kseg:
                print(f"    segment[{i+1}/{len(segments)}] original_text='{seg}' katakana_text='{kseg}'")
            kwargs = {}
            if length_scale is not None:
                kwargs['length'] = length_scale
            if pitch_scale is not None:
                kwargs['pitch'] = pitch_scale
            if assist_text is not None:
                kwargs['assist_text'] = assist_text
                kwargs['use_assist_text'] = True
            tts_sr, audio_i16 = tts_model.infer(
                kseg,
                speaker_id=speaker_id,style=speaker_style,
                **kwargs
            )

            if tts_sr != args.sr:
                # print(f"Warning: sample rate mismatch {tts_sr} != {args.sr}")
                audio_f32 = audio_i16.astype(np.float32) / 32768.0
                resampled_f32 = librosa.resample(audio_f32, orig_sr=tts_sr, target_sr=args.sr)
                audio_i16 = (resampled_f32 * 32768.0).clip(min=-32768, max=32767).astype(np.int16)
            total_audio_i16.append(audio_i16)
            s0 = total_length
            s1 = total_length + audio_i16.shape[0]
            t0 = round(s0 / args.sr, 3)
            t1 = round(s1 / args.sr, 3)
            print(f"    segment[{i+1}/{len(segments)}] samples {s0}:{s1} time {t0:.3f}:{t1:.3f} duration: {t1-t0:.3f}s text='{seg}'")
            timings.append( (s0, s1, seg) )
            total_length = s1

        audio_i16 = np.concatenate(total_audio_i16, axis=0)
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path_obj), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(args.sr)
            wf.writeframes(audio_i16.tobytes())
        num_samples = int(audio_i16.shape[0])
        duration_sec = num_samples / float(args.sr)
        info_path_obj = output_path_obj.with_suffix(".info")
        info_lines = [
            f"model-id: {model.id}",
            f"model-name: {model.name}",
            f"style: {speaker_style}",
            f"license: {dataset.license}",
            f"license-url: {dataset.license_url}",
            f"usage-terms: {dataset.usage_terms}",
            f"sample-rate: {args.sr}",
            f"total-text: {text}",
            f"total-samples: {num_samples}",
            f"total-duration-sec: {duration_sec:.3f}",
        ]
        for s0, s1, seg in timings:
            t0 = round(s0 / args.sr, 3)
            t1 = round(s1 / args.sr, 3)
            info_lines.append(f"segment: {s0}-{s1} time {t0}s-{t1}s duration {t1-t0}s text='{seg}'")
        info_path_obj.write_text("\n".join(info_lines) + "\n", encoding="utf-8")
        print(f"  done: {output_path_obj} monaural audio sampling rate:{args.sr} samples={num_samples} duration={duration_sec:.3f}(sec) ")

if __name__ == "__main__":
    sys.exit(main())
