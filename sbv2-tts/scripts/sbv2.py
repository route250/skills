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
    styles: dict[str, int] = field(default_factory=lambda: {'Neutral': 0})
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
            models=[ModelInfo(id='amitaro', name='あみたろ', gender='female',styles={'Neutral':0,'01':1,'02':2,'03':3, '04': 4})],
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
            models=[ModelInfo(id='koharune-ami', name='小春音アミ', gender='female',styles={
                'Neutral':0, 'るんるん':1, 'ささやきA(無声)': 2, 'ささやきB(有声)': 3, 'ノーマル':4, 'よふかし':5})],
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
                              styles={'Neutral':0, 'Angry':1, 'Disgust':2, 'Fear':3, 'Happy':4, 'Sad':5, 'suprise':6})],
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
                              styles={'Neutral':0, 'Angry':1, 'Disgust':2, 'Fear':3, 'Happy':4, 'Sad':5, 'suprise':6})],
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
                              styles={'Neutral':0, 'Angry':1, 'Disgust':2, 'Fear':3, 'Happy':4, 'Sad':5, 'suprise':6})],
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
                              styles={'Neutral':0, 'Angry':1, 'Disgust':2, 'Fear':3, 'Happy':4, 'Sad':5, 'suprise':6})],
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
            models=[ModelInfo(id='rinne', name='りんねおねんね', gender='female')],
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
                ModelInfo(id='amazinGood', name='amazinGood', spker_id=0, gender='female',description='落ち着いた女性の声',
                          styles={'Neutral':4, 'down':1, 'lol':2, 'ohmygod':3}),
                ModelInfo(id='calmCloud', name='calmCloud', spker_id=1, gender='female',description='落ち着いた女性の声',
                          styles={'Neutral':10, 'lol':5, 'question':6, 'down':7, 'hate': 8, 'ohmygod':9}),
                ModelInfo(id='coolcute', name='coolcute', spker_id=2, gender='female',description='落ち着いた女性の声',
                          styles={'Neutral':12, 'ohmygod':11, 'fine':13, 'sad':14}),
                ModelInfo(id='fineCrystal', name='fineCrystal', spker_id=3, gender='female',description='落ち着いた女性の声',
                          styles={'Neutral':18, 'fine':15, 'ohmygod':16, 'veryfine':17, 'sad':19}),
                ModelInfo(id='lightFire', name='lightFire', spker_id=4, gender='male',description='落ち着いた男性の声',
                          styles={'Neutral':22, 'question':20, 'hello':21, 'strong':23, 'lol':24}),
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
                              styles={'Neutral':0, 'Angry':1, 'Sad':2, 'Noisy':3, 'Clam': 4})],
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
                              styles={'Neutral':0, 'Happy':1, 'Sad':2, 'Angry':3})],
            license='配布者条件: 趣味の範囲で利用',
            license_url='https://hololivepro.com/terms/',
            usage_terms='モデルの取得や使い方は自由ですが、趣味の範囲で利用してください。詳細はカバー株式会社の二次創作ガイドライン（https://hololivepro.com/terms/）を確認してください。',
        ),
    ]

TEXT_PREPROCESS_RECOMMENDATION = (
    "【読み上げ前のテキスト前処理の推奨】\n"
    "- アルファベット表記は、できるだけカタカナに変換してから入力してください。\n"
    "- 読み間違いやすい漢字（例: 「方」「日」など）は、ひらがなに変換してから入力するほうが望ましいです。"
)

def print_text_preprocess_recommendation() -> None:
    print(TEXT_PREPROCESS_RECOMMENDATION)

def print_models_doc() -> int:
    data_list = get_datasets()
    print("# --modelオプションで指定できるモデルの一覧です。model-idを指定して下さい。")
    print("|model-id|model_name|gender|lang|description|")
    print("|---|---|---|---|---|")
    for ds in data_list:
        for model in ds.models:
            print(f"|{model.id}|{model.name}|{model.gender}|{model.language.name}|{model.description or ''}|")
    print("")
    return 0

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

def bbbb( dataset: DataSet, model: ModelInfo) -> None:
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


def print_model_info_doc(model_id: str) -> int:
    dataset, model = get_model_by_id(model_id)
    if dataset is None or model is None:
        print(f"モデルが見つかりません: {model_id}", file=sys.stderr)
        return 1

    bbbb(dataset, model)
    return 0

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
    parser.add_argument("--style", type=str, default=None, help="speaker style")
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

    if not args.model:
        dataset, model = get_default_model()
    else:
        dataset, model = get_model_by_id(args.model)

    if dataset is None or model is None:
        print(f"Model {args.model} not found. Using")
        return 1

    bbbb( dataset, model)

    print(f"num_jobs={len(texts)}")
    print(f"sample rate={args.sr}")
    print_text_preprocess_recommendation()

    language = model.language
    bert_models.load_model(language, SBV2_TOKENIZER_PATHS[language])
    bert_models.load_tokenizer(language, SBV2_TOKENIZER_PATHS[language])

    device: str|None = None
    tts_model = load_model(dataset, device=device)

    speaker_id:int = model.spker_id
    speaker_style:str = list(model.styles.keys())[0]
    speedScale:float = model.speedScale
    pitchOffset:float = model.pitchOffset

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

def dump():
    device = "cpu"
    data_list = get_datasets()
    for dataset in data_list:
        print(f"dataset model(s): {[model.id for model in dataset.models]}")
        tts_model = load_model(dataset, device=device)
        print("speakers information :")
        for k,v in tts_model.id2spk.items():
            print(f"id:{k} name:{v}")
        for k,v in tts_model.style2id.items():
            print(f"style_name={k} style_id={v}")

if __name__ == "__main__":
    dump() # main()
