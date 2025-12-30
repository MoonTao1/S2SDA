# Prompt-Guided Semantic Refinement for Cross-Weather Driving Saliency Adaptation

​	[note] We will release our complete code after the paper is **accepted** ✔️! Please look forward to it.🕓

## 📰 News

**[2025.8.27]** 🎈We have completed the training of the model and verified that it successfully enables the model to adapt to the rainy and night scenarios. 

**[2025.11.15]** 🎈We have conducted more detailed experiments, performing both qualitative and quantitative comparisons with other advanced methods, as well as a parameter comparison.

**[2025.12.2]** 🎈We will submit the article to ***ICME*** (IEEE International Conference on Multimedia and Expo 2026).😃

## ✨Model  

<div align="center">
<img src="fig\model.png" width=1000" height="auto" />
</div>


>The architecture of our model. The encoder adopts a dual-branch architecture. Cross-modal fusion is then performed to integrate the two. The unlabeled target data does not go through the decoder.

## ✨ Contributions

- **[PPIN] Pixel-level Prompt-Induced Normalization**  
  A pixel-level prompt-induced normalization module that integrates prompt-driven global weather priors with saliency-guided refinement to generate residual style offsets, enabling spatially adaptive feature modulation for zero-shot cross-weather adaptation.

- **[Mask-Guided] Backbone Modulation and Fusion**  
  A mask-guided modulation strategy where saliency masks steer early feature extraction toward weather-sensitive structural regions. The proposed Mask-Guidance Fusion Module further consolidates guided features, improving structural consistency under adverse weather.

- **[Benchmark] Cross-weather Evaluation Protocol**  
  We reorganize multiple public driving datasets according to fine-grained weather conditions and conduct extensive cross-weather evaluations, where **S2SDA consistently outperforms existing methods**.

## 💻 Dataset

<div align="center">
<img src="fig\dataset.png" width=60%" height="auto" />
</div>



>The datasets are organized as follows.
<div align="center">
<table>
<tr>
    <th>TrafficGaze</th>
    <th>DrFixD-rainy</th>
    <th>DrFixD-night</th>
  </tr>
  <tr>
    <td>
      ./TrafficGaze<br>
      &emsp;&emsp;|——fixdata<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata1.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata2.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata16.mat<br>
      &emsp;&emsp;|——trafficframe<br>
      &emsp;&emsp;|&emsp;&emsp;|——01<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|——000001.jpg<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——02<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——16<br>
      &emsp;&emsp;|——test.json<br>
      &emsp;&emsp;|——train.json<br>
      &emsp;&emsp;|——valid.json
    </td>
    <td>
      ./DrFixD-rainy<br>
      &emsp;&emsp;|——fixdata<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata1.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata2.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata16.mat<br>
      &emsp;&emsp;|——trafficframe<br>
      &emsp;&emsp;|&emsp;&emsp;|——01<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|——000001.jpg<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——02<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——16<br>
      &emsp;&emsp;|——test.json<br>
      &emsp;&emsp;|——train.json<br>
      &emsp;&emsp;|——valid.json
    </td>
        <td>
      ./DrFixD-night<br>
      &emsp;&emsp;|——fixdata<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata1.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata2.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata16.mat<br>
      &emsp;&emsp;|——trafficframe<br>
      &emsp;&emsp;|&emsp;&emsp;|——01<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|——000001.jpg<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——02<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——16<br>
      &emsp;&emsp;|——test.json<br>
      &emsp;&emsp;|——train.json<br>
      &emsp;&emsp;|——valid.json
    </td>
  </tr>
</table>
</div>


<div align="center">
<table>
<tr>
    <th>BDDA</th>
    <th>DADA</th>
  </tr>
  <tr>
   <td>
  ./BDDA<br>
  &emsp;&emsp;├── gazemap_frames/ <br>
  &emsp;&emsp;│&emsp;&emsp;├── 0002/ （0002 ~ 2017）<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;├── 0001.jpg<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;└── ... ... <br>
  &emsp;&emsp;│&emsp;&emsp;├── 0003/ <br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;└── 0001.jpg、0002.jpg、... ...<br>
  &emsp;&emsp;│&emsp;&emsp;└── ... ... <br>
  &emsp;&emsp;├── camera_frames/ <br>
  &emsp;&emsp;│&emsp;&emsp;├── 0002/ （0002 ~ 2017）<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;├── 0001.jpg<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;└── ... ... <br>
  &emsp;&emsp;│&emsp;&emsp;├── 0003/ <br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;└── 0001.jpg、0002.jpg、... ...<br>
  &emsp;&emsp;│&emsp;&emsp;└── ... ... <br>
    
  &emsp;&emsp;├── test_night.json <br>
  &emsp;&emsp;├── train_night.json <br>
  &emsp;&emsp;├── valid_night.json <br>
  &emsp;&emsp;└── ... ... 
</td>
  <td>
  ./DADA<br>
  &emsp;&emsp;|—— 01（1~52）<br>
  &emsp;&emsp;|&emsp;&emsp;|—— 001<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— fixation<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— images<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— maps<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— seg<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— semantic<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
  &emsp;&emsp;|&emsp;&emsp;|—— 002<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— fixation、images、maps、seg、semantic<br>
  &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
  &emsp;&emsp;|—— 02<br>
  &emsp;&emsp;|&emsp;&emsp;|—— + fixation/images/maps/seg/semantic<br>
  &emsp;&emsp;|—— ... ...）
   
  &emsp;&emsp;├── test_night.json <br>
  &emsp;&emsp;├── train_night.json <br>
  &emsp;&emsp;├── valid_night.json <br>
  &emsp;&emsp;└── ... ... 
</td>
  </tr>
</table>
</div>


## 🚀 Quantitative Analysis


>COMPARISON WITH OTHER METHODS
>Quantitative evaluation comparison of proposed model and the other methods from sunny to other weather on DADA.
<div align="center">
<img src="fig\DADA.png" width="80%" height="auto" />
</div>
<div align="center">
<img src="fig\DADA_mean.png" width="50%" height="auto" />
</div>
>Quantitative evaluation comparison of proposed model and the other methods from sunny to other weather on BDDA.
<div align="center">
<img src="fig\BDDA.png" width="60%" height="auto" />
</div>
>Quantitative evaluation comparison of proposed model and the other methods from sunny dataset TrafficGaze to rainy dataset DrFixD(rainy) and night dataset DrFixD(night).
<div align="center">
<img src="fig\Traffic.png" width="80%" height="auto" />
</div>




## 🚀Visualisation of intermediate results
>Qualitative evaluation comparison of proposed model and the other methods from sunny dataset TrafficGaze to rainy dataset DrFixD(rainy). The circles highlight objects/areas in the driving scene that disrupt the driver's attention.
<div align="center">
<img src="fig\visual.png" width="80%" height="auto" />
</div>

<div align="center">
<img src="fig\visual_ex.png" width="50%" height="auto" />
</div>




>Qualitative evaluation comparison of proposed model and the other methods from sunny dataset TrafficGaze to night dataset DrFixD(night). 
<div align="center">
<img src="fig\night.png" width="1200" height="auto" />
</div>
>Qualitative evaluation comparison of proposed model and the other methods from sunny dataset TrafficGaze to night dataset DrFixD(rainy). 
<div align="center">
<img src="fig\rainy.png" width="1200" height="auto" />
</div>

>Qualitative evaluation comparison of proposed model and the other methods from sunny to other weather on BDDA. 
<div align="center">
<img src="fig\BDDA_visual.png" width="1200" height="auto" />
</div>

>Qualitative evaluation comparison of proposed model and the other methods from sunny to other weather on DADA. 
<div align="center">
<img src="fig\DADA_night_vis.png" width="1200" height="auto" />
</div>

<div align="center">
<img src="fig\DADA_snowy.png" width="1200" height="auto" />
</div>



## 🛠️ Deployment **[🔁](#🔥Update)**
### 	Environment
  👉*If you wish to train with our model, please deploy the environment below.*
  ```python
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 	Run train 

​	👉*If you wish to train with our model, please use the proceeding steps below.*

1. Train our model.  You can use `--category` to switch datasets, which include `TrafficGaze`, `DrFixD-rainy`,`DrFixD-night` --b`  sets batch size, `--g  sets id of cuda.

```python
python main.py --network xxx --b 32 --g 0 --category xxx --root xxx
```


## ⭐️Cite

If you find this repository useful, please use the following BibTeX entry for citation.

```python
waiting accepted
```



































