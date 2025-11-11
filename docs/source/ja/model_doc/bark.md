<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Bark

## Overview

Bark は、[suno-ai/bark](https://github.com/suno-ai/bark) で Suno AI によって提案されたトランスフォーマーベースのテキスト読み上げモデルです。


Bark は 4 つの主要なモデルで構成されています。

- [`BarkSemanticModel`] ('テキスト'モデルとも呼ばれる): トークン化されたテキストを入力として受け取り、テキストの意味を捉えるセマンティック テキスト トークンを予測する因果的自己回帰変換モデル。
- [`BarkCoarseModel`] ('粗い音響' モデルとも呼ばれる): [`BarkSemanticModel`] モデルの結果を入力として受け取る因果的自己回帰変換器。 EnCodec に必要な最初の 2 つのオーディオ コードブックを予測することを目的としています。
- [`BarkFineModel`] ('微細音響' モデル)、今回は非因果的オートエンコーダー トランスフォーマーで、以前のコードブック埋め込みの合計に基づいて最後のコードブックを繰り返し予測します。
- [`EncodecModel`] からすべてのコードブック チャネルを予測したので、Bark はそれを使用して出力オーディオ配列をデコードします。

最初の 3 つのモジュールはそれぞれ、特定の事前定義された音声に従って出力サウンドを調整するための条件付きスピーカー埋め込みをサポートできることに注意してください。

### Optimizing Bark

Bark は、コードを数行追加するだけで最適化でき、**メモリ フットプリントが大幅に削減**され、**推論が高速化**されます。

#### Using half-precision

モデルを半精度でロードするだけで、推論を高速化し、メモリ使用量を 50% 削減できます。

```python
from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
```

#### Using 🤗 Better Transformer

Better Transformer は、内部でカーネル融合を実行する 🤗 最適な機能です。パフォーマンスを低下させることなく、速度を 20% ～ 30% 向上させることができます。モデルを 🤗 Better Transformer にエクスポートするのに必要なコードは 1 行だけです。

```python
model =  model.to_bettertransformer()
```

この機能を使用する前に 🤗 Optimum をインストールする必要があることに注意してください。 [インストール方法はこちら](https://huggingface.co/docs/optimum/installation)

#### Using CPU offload

前述したように、Bark は 4 つのサブモデルで構成されており、オーディオ生成中に順番に呼び出されます。言い換えれば、1 つのサブモデルが使用されている間、他のサブモデルはアイドル状態になります。

CUDA デバイスを使用している場合、メモリ フットプリントの 80% 削減による恩恵を受ける簡単な解決策は、アイドル状態の GPU のサブモデルをオフロードすることです。この操作は CPU オフロードと呼ばれます。 1行のコードで使用できます。

```python
model.enable_cpu_offload()
```

この機能を使用する前に、🤗 Accelerate をインストールする必要があることに注意してください。 [インストール方法はこちら](https://huggingface.co/docs/accelerate/basic_tutorials/install)

#### Combining optimization techniques

最適化手法を組み合わせて、CPU オフロード、半精度、🤗 Better Transformer をすべて一度に使用できます。

```python
from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# load in fp16
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)

# convert to bettertransformer
model = BetterTransformer.transform(model, keep_original_model=False)

# enable CPU offload
model.enable_cpu_offload()
```

推論最適化手法の詳細については、[こちら](https://huggingface.co/docs/transformers/perf_infer_gpu_one) をご覧ください。

### Tips

Suno は、多くの言語で音声プリセットのライブラリを提供しています [こちら](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)。
これらのプリセットは、ハブ [こちら](https://huggingface.co/suno/bark-small/tree/main/speaker_embeddings) または [こちら](https://huggingface.co/suno/bark/tree/main/speaker_embeddings)。

```python
>>> from transformers import AutoProcessor, BarkModel

>>> processor = AutoProcessor.from_pretrained("suno/bark")
>>> model = BarkModel.from_pretrained("suno/bark")

>>> voice_preset = "v2/en_speaker_6"

>>> inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

Bark は、非常にリアルな **多言語** 音声だけでなく、音楽、背景ノイズ、単純な効果音などの他の音声も生成できます。

```python
>>> # Multilingual speech - simplified Chinese
>>> inputs = processor("惊人的！我会说中文")

>>> # Multilingual speech - French - let's use a voice_preset as well
>>> inputs = processor("Incroyable! Je peux générer du son.", voice_preset="fr_speaker_5")

>>> # Bark can also generate music. You can help it out by adding music claude_notes around your lyrics.
>>> inputs = processor("♪ Hello, my dog is cute ♪")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

このモデルは、笑う、ため息、泣くなどの**非言語コミュニケーション**を生成することもできます。


```python
>>> # Adding non-speech cues to the input text
>>> inputs = processor("Hello uh ... [clears throat], my dog is cute [laughter]")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

オーディオを保存するには、モデル設定と scipy ユーティリティからサンプル レートを取得するだけです。

```python
>>> from scipy.io.wavfile import write as write_wav

>>> # save audio to disk, but first take the sample rate from the model config
>>> sample_rate = model.generation_config.sample_rate
>>> write_wav("bark_generation.wav", sample_rate, audio_array)
```

このモデルは、[Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe) および [Sanchit Gandhi (sanchit-gandhi)](https://github.com/sanchit-gandhi) によって提供されました。
元のコードは [ここ](https://github.com/suno-ai/bark) にあります。

## BarkConfig

[[autodoc]] BarkConfig
    - all

## BarkProcessor

[[autodoc]] BarkProcessor
    - all
    - __call__

## BarkModel

[[autodoc]] BarkModel
    - generate
    - enable_cpu_offload

## BarkSemanticModel

[[autodoc]] BarkSemanticModel
    - forward

## BarkCoarseModel

[[autodoc]] BarkCoarseModel
    - forward

## BarkFineModel

[[autodoc]] BarkFineModel
    - forward

## BarkCausalModel

[[autodoc]] BarkCausalModel
    - forward

## BarkCoarseConfig

[[autodoc]] BarkCoarseConfig
    - all

## BarkFineConfig

[[autodoc]] BarkFineConfig
    - all

## BarkSemanticConfig

[[autodoc]] BarkSemanticConfig
    - all
