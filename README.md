> **Overview**
> -
> _Reference_: https://pubs.acs.org/doi/10.1021/acsomega.4c03873.
> 
> ---
> <div style="text-align:justify;"> 
> Remaininguseful life (RUL) prediction is crucial for simplifying maintenance procedures and extending the life span of 
> aero engines. Therefore, research on RUL prediction methods for aero engines is increasingly gaining attention. 
> In particular, some existing deep neural networks based on multiscale features extraction have achieved certain results 
> in RUL predictions for aero-engines. However, these models often overlook two critical factors that affect RUL 
> prediction performance: (i) different time series data points have varying importance for RUL prediction, and (ii) the
> connections and similarities between different sensor data in both directions. This paper aims to extract valuable 
> multiscale features from raw monitoring data containing multiple sensor measurements, considering the aforementioned 
> factors, and leverage these features to enhance RUL prediction results.To this end, we propose a novel deep neural 
> network based on multiscale features extraction, named Multi-Scale Temporal-Spatial feature-based hybrid Deep 
> neural Network (MSTSDN). We conduct experiments using two aero-engine datasets, namely C-MAPSS and N-CMAPSS, to 
> evaluate RUL prediction performance of MSTSDN. Experimental results on C-MAPSS data set demonstrate that MSTSDN 
> achieves more accurate and timely RUL predictions compared to 12 existing deep neural networks specifically designed
> for predicting RUL of aero-engine, especially under multiple operational conditions and fault modes. And experimental 
> results on N-CMAPSS data set eventually indicate that MSTSDN can effectively track and fit with the actual RUL during 
> the engine degradation phase.
> </div>

> **PytorchModel**
> -
> ---
> Here is MSTSDN constructed using Pytorch (as an example on the DS07 subdataset of N-CMAPSS): <br>
> MSTSDNModel(<br>
  (pmtt): PMTTransformer(<br>
    (tf_modules): ModuleList(<br>
      (0-3): 4 x STFeatureExtractionLayer(<br>
        (multi_head_wa): WindowTransformerBlock(
          (attn_norm): LayerNorm((28,), eps=1e-05, elementwise_affine=True)<br>
          (mlp_norm): LayerNorm((28,), eps=1e-05, elementwise_affine=True)<br>
          (window_attn): WindowAttention(<br>
            (softmax): Softmax(dim=-1)<br>
            (q): Linear(in_features=28, out_features=32, bias=False)<br>
            (k): Linear(in_features=28, out_features=32, bias=False)<br>
            (v): Linear(in_features=28, out_features=32, bias=False)<br>
            (proj): Linear(in_features=32, out_features=28, bias=False)<br>
            (attn_dropout): Dropout(p=0.1, inplace=False)<br>
            (proj_dropout): Dropout(p=0.1, inplace=False)
          )
        )<br>
        (multi_head_swa): WindowTransformerBlock(<br>
          (attn_norm): LayerNorm((28,), eps=1e-05, elementwise_affine=True)<br>
          (mlp_norm): LayerNorm((28,), eps=1e-05, elementwise_affine=True)<br>
          (window_attn): WindowAttention(<br>
            (softmax): Softmax(dim=-1)<br>
            (q): Linear(in_features=28, out_features=32, bias=False)<br>
            (k): Linear(in_features=28, out_features=32, bias=False)<br>
            (v): Linear(in_features=28, out_features=32, bias=False)<br>
            (proj): Linear(in_features=32, out_features=28, bias=False)<br>
            (attn_dropout): Dropout(p=0.1, inplace=False)<br>
            (proj_dropout): Dropout(p=0.1, inplace=False)
          )<br>
          (mlp): MLP(<br>
            (linear1): Linear(in_features=28, out_features=44, bias=True)<br>
            (linear2): Linear(in_features=44, out_features=28, bias=True)<br>
            (act_1): GELU(approximate='none')<br>
            (act_2): GELU(approximate='none')<br>
            (dropout_0): Dropout(p=0.1, inplace=False)<br>
            (dropout_1): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )<br>
  (mbgcn): MBGraphConvolutionalNetwork(<br>
    (gsl_modules): ModuleList(<br>
      (0-3): 4 x AdaptiveGraphStructureLearningModule(<br>
        (linear1): Linear(in_features=30, out_features=23, bias=True)<br>
        (linear2): Linear(in_features=30, out_features=23, bias=True)<br>
        (adj_dropout): Dropout(p=0.1, inplace=False)<br>
        (act_1): GELU(approximate='none')<br>
        (act_2): GELU(approximate='none')
      )
    )<br>
    (bgc_networks): ModuleList(<br>
      (0-3): 4 x BGraphConvolutionalNetwork(<br>
        (bgc_layer): BidirectionalGraphConvolutionalLayer(<br>
          (dropout): Dropout(p=0.1, inplace=False)<br>
          (weight_f): Linear(in_features=30, out_features=30, bias=False)<br>
          (weight_b): Linear(in_features=30, out_features=30, bias=False)<br>
          (act): ReLU()
        )
      )
    )<br>
    (fusion_module): MFFusionModule(<br>
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )<br>
  (rn): RegressionNetwork(<br>
    (flatten): Flatten(start_dim=-2, end_dim=-1)<br>
    (mlp): Sequential(<br>
      (0): Linear(in_features=840, out_features=504, bias=True)<br>
      (1): ReLU()<br>
      (2): Dropout(p=0.75, inplace=False)<br>
      (3): Linear(in_features=504, out_features=151, bias=True)<br>
      (4): ReLU()<br>
      (5): Dropout(p=0.75, inplace=False)
    )<br>
    (regression_layer): Linear(in_features=151, out_features=1, bias=True)
  )
).<br>

> **Datasets**
> -
> _Reference_: 1. https://ieeexplore.ieee.org/document/4711414; 2. https://www.mdpi.com/2306-5729/6/1/5.
> 
> ---
> Here are the data source links for C-MAPSS and N-CMAPSS: <br>
> 1.https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository ;<br>
> 2.https://data.nasa.gov/Aeorspace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6 ;<br>
> 3.https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip ;<br>
> 4.https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip .<br>
> Information and files on the relevant datasets can be accessed and downloaded through these links.

> **Environment**
> -
> ---
> The following is the experimental environment used:<br>
> GPU: NVIDIA GeForce RTX 3070;<br>
> CPU: AMD Ryzen 7 5800;<br>
> Pytorch: version==2.2.1+cu118;<br>
> Python: version==3.11.8.<br> 

> **Citation**
> -
> ---
> If the project is helpful to your work, please cite the following paper:<br>
> **Multi-Scale Temporal-Spatial Feature-Based Hybrid Deep Neural Network for Remaining Useful Life Prediction of Aero-Engine<br>
> Zhaofeng Liu, Xiaoqing Zheng, Anke Xue and Ming Ge <br>
> ACS Omega 2024 <br>
> DOI 10.1021/acsomega.4c03873** 
