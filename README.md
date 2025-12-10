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
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Train (video/frame)</th>
      <th>Valid (video/frame)</th>
      <th>Test (video/frame)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TrafficGaze</td>
      <td>49080</td>
      <td>6655</td>
      <td>19135</td>
    </tr>
    <tr>
      <td>DrFixD(rainy)</td>
      <td>52291</td>
      <td>9816</td>
      <td>19154</td>
    </tr>
    <tr>
      <td>DrFixD(night)</td>
      <td>41987</td>
      <td>10319</td>
      <td>14773</td>
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
>COMPARISON WITH OTHER METHODS FROM TraffiicGaze TO DRFIXD(RAINY)

<h3 align="center">Generalization Performance from DADA<sub>sunny</sub> → DADA<sub>t</sub></h3>

<!-- =================== 第一部分：RAINY + SNOWY =================== -->
<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Model</th>
      <th colspan="6">RAINY</th>
      <th colspan="6">SNOWY</th>
    </tr>
    <tr>
      <th>AUC_B&#8593;</th><th>AUC_J&#8593;</th><th>NSS&#8593;</th><th>CC&#8593;</th><th>SIM&#8593;</th><th>KLD&#8595;</th>
      <th>AUC_B&#8593;</th><th>AUC_J&#8593;</th><th>NSS&#8593;</th><th>CC&#8593;</th><th>SIM&#8593;</th><th>KLD&#8595;</th>
    </tr>
  </thead>

  <tbody>

    <!-- Traditional -->
    <tr>
      <td rowspan="5">Traditional</td>
      <td>CDNN</td><td>0.866</td><td>0.920</td><td>3.048</td><td>0.469</td><td>0.288</td><td>1.738</td>
                     <td>0.743</td><td>0.852</td><td>1.396</td><td>0.265</td><td>0.216</td><td>2.249</td>
    </tr>

    <tr>
      <td>CPFE</td><td>0.875</td><td>0.921</td><td>2.984</td><td>0.462</td><td>0.289</td><td>1.735</td>
                   <td>0.735</td><td>0.841</td><td>1.328</td><td>0.255</td><td>0.212</td><td>2.329</td>
    </tr>

    <tr>
      <td>Transal</td><td>0.841</td><td>0.905</td><td>2.555</td><td>0.408</td><td>0.266</td><td>1.929</td>
                     <td><strong>0.813</strong></td><td>0.860</td><td>1.505</td><td>0.280</td><td>0.207</td><td>2.148</td>
    </tr>

    <tr>
      <td>SCOUT</td><td>0.868</td><td>0.914</td><td>2.720</td><td>0.430</td><td>0.271</td><td>1.840</td>
                   <td>0.734</td><td>0.840</td><td>1.296</td><td>0.248</td><td>0.209</td><td>2.373</td>
    </tr>

    <tr>
      <td>STDENet</td><td>0.877</td><td>0.926</td><td>3.170</td><td>0.489</td><td>0.318</td><td>1.647</td>
                     <td>0.751</td><td>0.854</td><td>1.469</td><td>0.276</td><td>0.226</td><td>2.265</td>
    </tr>

    <!-- UDA -->
    <tr>
      <td rowspan="6">UDA</td>
      <td>MT</td><td>0.874</td><td>0.917</td><td>2.844</td><td>0.444</td><td>0.281</td><td>1.785</td>
                 <td>0.782</td><td><strong>0.862</strong></td><td>1.583</td><td><strong>0.298</strong></td><td>0.228</td><td><strong>2.128</strong></td>
    </tr>

    <tr>
      <td>DANN</td><td>0.858</td><td>0.912</td><td>2.720</td><td>0.425</td><td>0.264</td><td>1.862</td>
                   <td>0.785</td><td>0.848</td><td>1.538</td><td>0.251</td><td>0.169</td><td>2.406</td>
    </tr>

    <tr>
      <td>DRCN</td><td>0.815</td><td>0.893</td><td>2.556</td><td>0.396</td><td>0.240</td><td>2.021</td>
                   <td>0.737</td><td>0.814</td><td>1.132</td><td>0.221</td><td>0.190</td><td>2.471</td>
    </tr>

    <tr>
      <td>HD2S</td><td>0.843</td><td>0.867</td><td>1.809</td><td>0.290</td><td>0.177</td><td>2.537</td>
                   <td>0.536</td><td>0.529</td><td>0.243</td><td>0.052</td><td>0.090</td><td>2.998</td>
    </tr>

    <tr>
      <td>AT</td><td>0.871</td><td>0.915</td><td>2.762</td><td>0.434</td><td>0.273</td><td>1.830</td>
                 <td>0.740</td><td>0.848</td><td>1.406</td><td>0.265</td><td>0.223</td><td>2.295</td>
    </tr>

    <tr>
      <td>MHDAN</td><td>0.850</td><td>0.900</td><td>2.430</td><td>0.388</td><td>0.234</td><td>2.029</td>
                    <td>0.771</td><td>0.834</td><td>1.406</td><td>0.266</td><td>0.199</td><td>2.327</td>
    </tr>

    <!-- ZSDA -->
    <tr>
      <td rowspan="3">ZSDA</td>
      <td>PODA</td><td>0.868</td><td>0.925</td><td>3.226</td><td>0.497</td><td><strong>0.327</strong></td><td>1.626</td>
                   <td>0.732</td><td>0.848</td><td>1.436</td><td>0.269</td><td>0.227</td><td>2.301</td>
    </tr>

    <tr>
      <td>ULDA</td><td>0.876</td><td>0.915</td><td>2.849</td><td>0.447</td><td>0.282</td><td>1.810</td>
                   <td>0.753</td><td>0.831</td><td>1.391</td><td>0.266</td><td>0.218</td><td>2.476</td>
    </tr>

    <tr>
      <td><strong>Ours</strong></td><td><strong>0.877</strong></td><td><strong>0.929</strong></td><td><strong>3.293</strong></td><td><strong>0.506</strong></td><td>0.326</td><td><strong>1.598</strong></td>
                   <td>0.756</td><td>0.857</td><td><strong>1.608</strong></td><td><strong>0.298</strong></td><td><strong>0.241</strong></td><td>2.188</td>
    </tr>

  </tbody>
</table>




<div align="center">
<img src="pic\vis_rainy_00.png" width="800" height="auto" />
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
<img src="pic\night_00.png" width="1200" height="auto" />
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









