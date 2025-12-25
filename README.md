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
<img src="pic\vis_00.png" width=600" height="auto" />
</div>
>- 🟥 The red square represents elements that are not easily perceived during driving yet have the potential to cause accidents.
 - 🟡 Indicates distracting factors unrelated to the driving task.


## 💻Dataset

<div align="center">
<img src="pic\dataset.png" width="700" height="auto" />
</div>

<div align="center">
  <img src="pic/dataset.png" width="700" height="auto" />
</div>

<br>

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Weather</th>
        <th>Train</th>
        <th>Valid</th>
        <th>Test</th>
      </tr>
    </thead>
    <tbody>
      <!-- DADA -->
      <tr>
        <td rowspan="5"><b>DADA</b></td>
        <td>sunny</td>
        <td>24052</td>
        <td>3492</td>
        <td>5965</td>
      </tr>
      <tr>
        <td>rainy</td>
        <td>12994</td>
        <td>2887</td>
        <td>5778</td>
      </tr>
      <tr>
        <td>night</td>
        <td>19965</td>
        <td>2566</td>
        <td>14687</td>
      </tr>
      <tr>
        <td>foggy</td>
        <td>848</td>
        <td>187</td>
        <td>378</td>
      </tr>
      <tr>
        <td>snowy</td>
        <td>544</td>
        <td>121</td>
        <td>242</td>
      </tr>

      <!-- BDDA -->
      <tr>
        <td rowspan="3"><b>BDDA</b></td>
        <td>sunny</td>
        <td>26883</td>
        <td>10858</td>
        <td>16375</td>
      </tr>
      <tr>
        <td>rainy</td>
        <td>3212</td>
        <td>553</td>
        <td>1168</td>
      </tr>
      <tr>
        <td>night</td>
        <td>7592</td>
        <td>3939</td>
        <td>3504</td>
      </tr>

      <!-- Others -->
      <tr>
        <td><b>TrafficGaze</b></td>
        <td>sunny</td>
        <td>49080</td>
        <td>6655</td>
        <td>19135</td>
      </tr>
      <tr>
        <td><b>DrFixD (night)</b></td>
        <td>night</td>
        <td>41987</td>
        <td>10319</td>
        <td>14773</td>
      </tr>
      <tr>
        <td><b>DrFixD (rainy)</b></td>
        <td>rainy</td>
        <td>52291</td>
        <td>9816</td>
        <td>19154</td>
      </tr>
    </tbody>
  </table>
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

<div align="center">
<img src="fig\rainy.png" width="1200" height="auto" />
</div>


<div align="center">
<img src="fig\BDDA_visual.png" width="1200" height="auto" />
</div>

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














