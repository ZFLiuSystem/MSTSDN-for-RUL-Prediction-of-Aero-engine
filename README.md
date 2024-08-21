Remaining useful life (RUL) prediction is crucial for simplifying maintenance procedures and extending the lifespan of aero-engines. 
Therefore, research on RUL prediction methods for aero-engines is increasingly gaining attention. To this end, a novel deep neural network 
based on multi-scale feature extraction, named Multi-Scale Temporal-Spatial feature-based hybrid Deep neural Network (MSTSDN). 
The experiments were conducted by using two aero-engine datasets, namely C-MAPSS and N-CMAPSS, to evaluate RUL prediction performance of MSTSDN. 
The repository contains the datasets used, the code to construct MSTSDN,  and the code to train and test the model.


The following is the experimental environment used:
GPU: NVIDIA GeForce RTX 3070.
CPU: AMD Ryzen 7 5800.
Pytorch: version==2.2.1+cu118.
Python: version==3.11.8.


Finally, here are the data source links for C-MAPSS and N-CMAPSS:
1.https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository.
2.https://data.nasa.gov/Aeorspace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6.
3.https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip.
4.https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip.
