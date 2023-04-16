# [Source-Free Cross-Domain State of Charge Estimation of Lithium-ion Batteries at Different Ambient Temperatures](https://ieeexplore.ieee.org/document/10058040)
Liyuan Shen; Jingjing Li; Lin Zuo; Lei Zhu; Heng Tao Shen  
Abstract:Machine learning methods for state of charge (SOC) estimation of lithium-ion batteries (LiBs) face the problem of domain shift. Varying conditions such as different ambient temperatures can cause performance degradation of the estimators due to data distribution discrepancy. Some transfer learning methods have been utilized to tackle the problem. At real-time transfer, the source model is supposed to keep updating itself online. In the process, source domain data are usually absent because the storage and acquisition of all historical running data can involve violating the privacy of users. However, existing methods require coexistence of source and target samples. In this paper, we discuss a more difficult yet more practical source-free setting where there are only the models pre-trained in source domain and limited target data can be available. To address the challenges of the absence of source data and distribution discrepancy in cross-domain SOC estimation, we propose a novel source-free temperature transfer network (SFTTN), which can mitigate domain shift adaptively. In this paper, cross-domain SOC estimation under source-free transfer setting is discussed for the first time. To this end, we define an effective approach named minimum estimation discrepancy (MED), which attempts to align domain distributions by minimizing the estimation discrepancy of target samples. Extensive transfer experiments and online testing at fixed and changing ambient temperatures are performed to verify the effectiveness of SFTTN. The experiment results indicate that SFTTN can achieve robust and accurate SOC estimation at different ambient temperatures under source-free scenario.
# Usage
* conda environment   
```
conda env create -f env.yaml
```
* Dataset  

more dataset for LIBs can be downloaded from [HERE](https://docs.google.com/spreadsheets/d/10w5yXdQtlQjTTS3BxPP233CiiBScIXecUp2OQuvJ_JI/edit#gid=0)
* Data processing  

put your data fold in ```normalized_data/``` and run this code  
```
python normalized_data/dataprocess.py
```
* To pretrain source model (including two different estimators)    
```
python run.py --mode pretrain --mkdir [your_folder] --source_data_path [] --source_temp [] --epochs --batch_size
```
(check run.py for more arguments)  
The model is saved in ```run/your_folder/saved_model/best.pt```
* Pseudo label    

Use pre-trained source model to generate pseudo labels for target data:    
```
python pseudo.py --temp --model --file
```
* To transfer a model  
```
python run.py --mode train --mkdir [] --source_data_path --source_temp --target_data_path --target_temp --epochs --batch_size
```
(check run.py for more arguments)   
* models  

We have provided pretrained models and models retrained only with limitted target labels for five temperatures of Panasonic 18650PF dataset in folder "models" for comparison.
* To test a model  
```
python run.py --mode test --mkdir [] --test_set [] --target_temp []
```
