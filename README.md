# Driving Fixation Prediction for Clear-to-Adverse Weather Scenes via Adversarial Unsupervised Domain Adaptation

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

<div align="center">
<img src="fig\vis_00.png" width=600" height="auto" />
</div>
>- 🟥 The red square represents elements that are not easily perceived during driving yet have the potential to cause accidents.
 - 🟡 Indicates distracting factors unrelated to the driving task.


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
  &emsp;&emsp;├── gazemap_frames/ （凝视图帧文件夹）<br>
  &emsp;&emsp;│&emsp;&emsp;├── 0002/ （视频文件夹：0002 ~ 2017）<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;├── 0001.jpg<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;├── 0002.jpg<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;└── ... ... （后续编号jpg图片）<br>
  &emsp;&emsp;│&emsp;&emsp;├── 0003/ （视频文件夹编号）<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;└── 0001.jpg、0002.jpg、... ...（同0002结构）<br>
  &emsp;&emsp;│&emsp;&emsp;└── ... ... （更多视频文件夹，编号至2017）<br>
  &emsp;&emsp;├── camera_frames/ （相机图帧文件夹）<br>
  &emsp;&emsp;│&emsp;&emsp;├── 0002/ （视频文件夹：0002 ~ 2017）<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;├── 0001.jpg<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;├── 0002.jpg<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;└── ... ... （后续编号jpg图片）<br>
  &emsp;&emsp;│&emsp;&emsp;├── 0003/ （视频文件夹编号）<br>
  &emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;└── 0001.jpg、0002.jpg、... ...（同0002结构）<br>
  &emsp;&emsp;│&emsp;&emsp;└── ... ... （更多视频文件夹，编号至2017）<br>
  &emsp;&emsp;├── test.json （测试集天气相关配置文件）<br>
  &emsp;&emsp;├── train.json （训练集天气相关配置文件）<br>
  &emsp;&emsp;├── valid.json （验证集天气相关配置文件）<br>
  &emsp;&emsp;└── ... ... （其他不同天气类型的json文件）
</td>
  <td>
  ./DADA<br>
  &emsp;&emsp;|—— 01（视频编号，1~52）<br>
  &emsp;&emsp;|&emsp;&emsp;|—— 001（子视频编号，按需层级展示）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— fixation（注视点文件夹）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 002.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...（编号png图片）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— images（图片文件夹）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 002.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...（编号png图片）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— maps（映射文件夹）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 002.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...（编号png图片）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— seg（分割文件夹）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 002.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...（编号png图片）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— semantic（语义文件夹）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 001.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— 002.png<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...（编号png图片）<br>
  &emsp;&emsp;|&emsp;&emsp;|—— 002（子视频编号）<br>
  &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— fixation、images、maps、seg、semantic（同上述结构）<br>
  &emsp;&emsp;|&emsp;&emsp;|—— ... ...（更多子视频编号）<br>
  &emsp;&emsp;|—— 02（视频编号）<br>
  &emsp;&emsp;|&emsp;&emsp;|—— 子视频编号 + fixation/images/maps/seg/semantic（同上述结构）<br>
  &emsp;&emsp;|—— ... ...（视频编号3~52，均遵循上述目录结构）
</td>
写法二：清晰
  </tr>
</table>
</div>


## 🚀 Quantitative Analysis

<div align="center">
<img src="fig\visual.png" width="800" height="auto" />
</div>





<div align="center">
<img src="fig\vis_rainy.png" width="800" height="auto" />
</div>





>COMPARISON WITH OTHER METHODS FROM TraffiicGaze TO DRFIXD(NIGHT)

<div align="center">
  <table border="1" style="margin: 0 auto;">
    <thead>
      <tr>
        <th>Model</th>
        <th>AUC_B↑</th>
        <th>AUC_J↑</th>
        <th>NSS↑</th>
        <th>CC↑</th>
        <th>SIM↑</th>
        <th>KLD↓</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>CDNN</td>
        <td>0.7618</td>
        <td>0.8591</td>
        <td>1.8446</td>
        <td>0.5071</td>
        <td>0.4002</td>
        <td>1.2286</td>
      </tr>
      <tr>
        <td>CPFE</td>
        <td>0.7992</td>
        <td>0.9111</td>
        <td>3.2687</td>
        <td>0.6979</td>
        <td>0.5737</td>
        <td>0.7530</td>
      </tr>
      <tr>
        <td>TransalNet</td>
        <td>0.8404</td>
        <td>0.9390</td>
        <td>3.2687</td>
        <td>0.6979</td>
        <td>0.5737</td>
        <td>0.7530</td>
      </tr>
      <tr>
        <td>SCOUT</td>
        <td>0.8269</td>
        <td>0.9122</td>
        <td>2.6843</td>
        <td>0.6091</td>
        <td>0.4960</td>
        <td>1.0103</td>
      </tr>
      <tr>
        <td>STDENet</td>
        <td>0.8676</td>
        <td>0.9345 </td>
        <td>3.1000 </td>
        <td>0.7105</td>
        <td>0.5687</td>
        <td>0.7531</td>
      </tr>
      <tr>
        <td>MT</td>
        <td>0.8212</td>
        <td>0.9204</td>
        <td>3.0749</td>
        <td>0.6752</td>
        <td>0.5474 </td>
        <td>0.8865</td>
      </tr>
      <tr>
        <td>DANN</td>
        <td>0.8342</td>
        <td>0.9021</td>
        <td>2.4349</td>
        <td>0.5414</td>
        <td>0.4024</td>
        <td>1.2041</td>
      </tr>
      <tr>
        <td>DRCN</td>
        <td>0.8224</td>
        <td>0.9174</td>
        <td>2.6752</td>
        <td>0.6258</td>
        <td>0.5099</td>
        <td>0.9422</td>
      </tr>
      <tr>
        <td>HD2S</td>
        <td>0.8699</td>
        <td>0.9113</td>
        <td>2.3358</td>
        <td>0.5774</td>
        <td>0.4733</td>
        <td>1.0028</td>
      </tr>
      <tr>
        <td>AT</td>
        <td>0.8650</td>
        <td>0.9385</td>
        <td>3.0265</td>
        <td>0.6971</td>
        <td>0.5696</td>
        <td>0.7431</td>
      </tr>
      <tr>
        <td>MHDAN</td>
        <td><strong>0.8763</strong></td>
        <td>0.9109</td>
        <td>2.5688 </td>
        <td>0.6403</td>
        <td>0.4701</td>
        <td>0.9794</td>
      </tr>
      <tr>
        <td>Ours</td>
        <td><strong>0.8763</strong></td>
        <td><strong>0.9401</strong></td>
        <td><strong>3.3666</strong></td>
        <td><strong>0.7498</strong></td>
        <td><strong>0.5976</strong></td>
        <td><strong>0.6528</strong></td>
      </tr>
    </tbody>
  </table>
</div>

<div align="center">
<img src="fig\visual.png" width="1200" height="auto" />
</div>


## 🚀Visualisation of intermediate results
>Qualitative evaluation comparison of proposed model and the other methods from sunny dataset TrafficGaze to rainy dataset DrFixD(rainy). The circles highlight objects/areas in the driving scene that disrupt the driver's attention.

<div align="center">
<img src="fig\visual_ex.png" width="800" height="auto" />
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





















