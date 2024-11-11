>**Overview**
<div style="text-align: justify;">
This project aims to fine-tune the GPT-2 model provided by HuggingFace using the LoRA algorithm to adapt it for daily English conversation. The dataset utilized for this purpose is the DailyDialog dataset available on Kaggle. Through a series of training and adjustments, I have successfully constructed a Chatbot capable of engaging daily conversation in English. Currently, all datasets, model, and files and codes related to training and testing used in this project have been open-sourced for public access and use.</div>

>**KeyPoints**<br>
* **Algorithm:** <div style="text-align: justify;">LoRA, an efficient algorithm for model fine-tuning, particularly suitable for natural language processing tasks such as dialogue generation.</div>
---
* **Model:** <div style="text-align: justify;">GPT-2, provided by HuggingFace, is a pre-trained language model based on the Transformer architecture, adept at generating coherent text.</div>
---
* **Dataset:** <div style="text-align: justify;">DailyDialog, sourced from Kaggle, contains a large amount of daily dialogue data, making it highly suitable for training chatbots.</div>
---
* **Outcome:** <div style="text-align: justify;">A Chatbot capable of engaging in daily English conversation.</div>
---
* **OpenSource:**<div style="text-align: justify;"> All project resources (datasets, model, codes, etc.) have been made public, facilitating learning and reuse. In summary, this project displays how to build a usable chatbot using limited resources and algorithms.</div>

>**GPT-2**<br>

![GptForChat.jpg](Graph_%2FGptForChat.jpg)
<div style="text-align: justify;">
GPT-2 (Generative Pre-training Transformer 2.0), introduced by OpenAI in 2019 as the second generation of generative pre-training model, is an auto-regressive Decoder-only Transformer. GPT-2 is based on the auto-regression mechanism, which predicts the unknown subsequent text based on the known preceding text. This mechanism enables GPT-2 to maintain coherence and consistency when generating text. Additionally, GPT-2 adopts a Decoder-only structure, utilizing only the decoder part of the Transformer. This structure makes GPT-2 more suitable for generative tasks such as text generation and dialogue simulation.</div>

![CasualAttention.jpg](Graph_%2FCasualAttention.jpg) <br>
<div style="text-align: justify;">The causal self-attention mechanism within GPT-2 ensures that the model strictly follows the temporal order when generating text, meaning it can only use past and current tokens to predict future tokens. This mechanism relies on the masking operation of the attention matrix. This causal self-attention bears similarities to traditional language models, which also predict future tokens based on past tokens to generate coherent text sequences. GPT-2 adheres to the style of traditional language models while incorporating the Transformer architecture, achieving more efficient and accurate text generation.</div>

**Reference:** https://paperswithcode.com/paper/language-models-are-unsupervised-multitask
>**LoRA**<br>


![LoraTuning.jpg](Graph_%2FLoraTuning.jpg)
<div style="text-align: justify;">Pre-trained large language models, such as the GPT series, often face issues of high computational power consumption and long training times when undergoing full fine-tuning for downstream tasks due to their vast number of parameters. To address this challenge, LoRA (Low-Rank Adaptation) emerges as a highly effective method.<br>
<br>As an efficient fine-tuning strategy, the core concept of LoRA lies in reducing the amount of parameters to be trained through low-rank decomposition. Specifically, instead of directly fine-tuning all the parameters of the pre-trained model, LoRA combines the low-rank updates of the model parameters with the original parameters to achieve fine-tuning. This approach not only significantly reduces the number of parameters that need to be trained but also lowers the GPU memory footprint, making fine-tuning of large models feasible with limited resources.<br>
<br>In fact, LoRA achieves efficient fine-tuning without compromising the model's performance. According to relevant research and experimental validation, models fine-tuned with LoRA perform comparably to those fully fine-tuned on downstream tasks and can even achieve better results in some cases. This advantage has made LoRA widely popular in practical applications.</div>

**Reference:** https://paperswithcode.com/paper/lora-low-rank-adaptation-of-large-language

>**Environment**
* **Software:**<div style="text-align: justify;">IDE==Pycharm 2023.2; Python==3.11; Pytorch==2.2.1; transformers==4.42.3; peft==0.10.0.</div>
---
* **Hardware:**<div style="text-align: justify;">CPU==AMD Ryzen 7 5800H; <br>GPU==NVIDIA GeForce RTX3070 Laptop (8GB); <br>RAM==Micron Technology DDR4 (16GB).</div>
>**Training&Test**
* **Model:**<br>
ChatModel(<br>
  (gpt_backbone): PeftModelForFeatureExtraction(<br>
    (base_model): LoraModel(<br>
      (model): GPT2Model(<br>
        (wte): Embedding(50257, 768)<br>
        (wpe): Embedding(1024, 768)<br>
        (drop): Dropout(p=0.1, inplace=False)<br>
        (h): ModuleList(<br>
          (0-11): 12 x GPT2Block(<br>
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)<br>
            (attn): GPT2SdpaAttention(<br>
              (c_attn): lora.Linear(<br>
                (base_layer): Conv1D()<br>
                (lora_dropout): ModuleDict(<br>
                  (default): Dropout(p=0.1, inplace=False)<br>
                )<br>
                (lora_A): ModuleDict(<br>
                  (default): Linear(in_features=768, out_features=8, bias=False)<br>
                )<br>
                (lora_B): ModuleDict(<br>
                  (default): Linear(in_features=8, out_features=2304, bias=False)<br>
                )<br>
                (lora_embedding_A): ParameterDict()<br>
                (lora_embedding_B): ParameterDict()<br>
              )<br>
              (c_proj): Conv1D()<br>
              (attn_dropout): Dropout(p=0.1, inplace=False)<br>
              (resid_dropout): Dropout(p=0.1, inplace=False)<br>
            )<br>
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)<br>
            (mlp): GPT2MLP<br>
              (c_fc): Conv1D()<br>
              (c_proj): Conv1D()<br>
              (act): NewGELUActivation()<br>
              (dropout): Dropout(p=0.1, inplace=False)<br>
            )
          )
        )<br>
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)<br>
      )
    )
  )<br>
  (head): Linear(in_features=768, out_features=50257, bias=False)
  (dropout): Dropout(p=0.1, inplace=False)<br>
)
---
* **TrainingPart:**
![Traninig_.png](Graph_%2FTraninig_.png)
---
* **TestPart:**
![Test_.png](Graph_%2FTest_.png)
---
* **Notice:**
There are some compatibility issues between the versions of Pytorch and transformers, so Pycharm throws some warnings during training and test.