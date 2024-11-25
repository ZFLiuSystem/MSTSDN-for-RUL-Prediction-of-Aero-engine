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
