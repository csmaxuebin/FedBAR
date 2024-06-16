# This code is the source code implementation for the paper "FedBAR: Block Level Data Augmentation Privacy Protection method of Federated Learning."



## Abstract
![DP-CUDA框架图](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/1.png)
Federal Learning (FL) is a privacy-centric distributed machine learning framework, which mitigates privacy risks by sharing model updates rather than data. However, recent research indicates that sharing model updates cannot exempt FL from the threat of inference attacks. This study delves into the reasons for data leakage in FL under heterogeneous environments, revealing that attack Algorithms rely on prior knowledge and auxiliary information drawn from gradients for attacks. Based on these findings, we propose a block-level data augmentation privacy protection method for FL, addressing the reliance of attack algorithms on prior knowledge and auxiliary information. By undermining the validity of prior knowledge and preventing attack algorithms from utilizing auxiliary information to reconstruct private samples, the privacy risk can be reduced while maintaining the performance of FL. This is achieved by applying data augmentation techniques that incorporate privacy protection capabilities to the data representation. This paper conducts reconstruction attack experiments and, without compromising accuracy, significantly decreases the correlation between the data representations restored by the attack algorithms and the true data representations. This enhances the privacy protection capability of FL.


# Experimental Environment

```
- breaching==0.1.2
- calmsize==0.1.3
- h5py==3.8.0
— opacus==1.4.0
- Pillow==9.2.0
- scikit-learn==1.2.2
- sklearn==0.0.post1
- torch==2.0.0
- torchvision~=0.15.1+cu117
- ujson==5.7.0
- numpy==1.23.2
- scipy==1.8.1
- matplotlib==3.5.2
```

## Datasets

`Cifar10,Celeba`


## Experimental Setup


**Hyperparameters:**

-   Training is conducted over 500 rounds with 10 clients participating. Each client executes 1 epoch per round.
-   The learning rate is set at 0.005, and the batch size is 32.
-   An adaptive clipping threshold is initialized at the 0.5 quantile, with an adaptive learning rate for clipping set at 0.2.
-   Each client’s data, both in terms of the number of samples and the classes of those samples, is randomly determined and follows a Dirichlet distribution to simulate a heterogeneous data environment typical in real-world scenarios.

**Attack Models:** Four types of reconstruction attacks are tested:

1.  **Deep Leakage from Gradients (DLG):** This method calculates the L2 loss between generated and real gradients and optimizes the generated samples using an optimizer.
2.  **Improved DLG (iDLG):** Builds on DLG by utilizing prior knowledge of labels.
3.  **Inversion of Gradients (IG):** Uses cosine distance as a loss metric, incorporating prior knowledge about the overall variance of gradients.
4.  **Generative Gradient Leakage (GGL):** Employs a GAN trained on a public dataset to generate samples.

All attack models optimize generated samples using the Adam optimizer.

**Privacy-Preserving Methods:** Comparison is made between the proposed method and four privacy-preserving techniques:

1.  **Differential Privacy (DP):** Applies Gaussian noise with σ=1 to the gradients to ensure privacy.
2.  **Gradient Clipping:** Uses L2 norm clipping with a maximum norm of 4.
3.  **Gradient Sparsification:** Prunes gradients to achieve a sparsity of 90%.
4.  **Soteria:** Prunes gradients based on information extraction, targeting gradients with the top 80% extraction quantity.

**Evaluation Metrics:**

-   **Model Performance:** Model accuracy and loss are used as primary metrics.
-   **Privacy Protection:** Privacy is assessed visually and through several quantitative metrics:
    1.  **Image Mean Squared Error (MSE-I):** Pixel-level MSE between the original and reconstructed samples.
    2.  **Peak Signal-to-Noise Ratio (PSNR):** Ratio of maximum pixel value to MSE.
    3.  **Learned Perceptual Image Patch Similarity (LPIPS):** Measures perceptual similarity using a pre-trained VGG model.
    4.  **Data Representation MSE (MSE-R):** Similar to image MSE, but calculates the error between data representations of the original and reconstructed samples.
## Python Files
-   **env_cuda_102.yaml and env_cuda_116.yaml**: These are likely YAML files specifying configurations for creating Python environments with specific packages and dependencies. The names suggest configurations for environments compatible with different CUDA versions, which would be useful for GPU-accelerated computing tasks.
    
-   **examples.sh**: This shell script includes a series of commented-out commands for running machine learning experiments using different algorithms and configurations on various datasets like MNIST, CIFAR10, CIFAR100, and others. It sets up environment configurations, initiates training processes, and manages output logging, showcasing a structured approach to executing numerous experiments in a batch mode.
    
-   **get_mean_std.py**: This Python script is probably used to calculate the mean and standard deviation of datasets. These statistics are often required for normalizing datasets before training machine learning models to improve model performance and stability during training.
    
-   **log_clean_mixx_mixrep_aby.txt**: This text file contains logs from running machine learning experiments, detailing parameters like batch sizes, learning rates, and results across training rounds. Such logs are essential for monitoring model performance and debugging during experimental runs.
    
-   **main.py**: This is likely the main Python script that orchestrates the execution of federated learning experiments. It might handle the setup of experiments, including model initialization, data distribution among clients, and the orchestration of training rounds.
    
-   **test.py**: Typically, a test script in Python is used for running a suite of automated tests that check the correctness of code or model behavior against expected outcomes. In the context of machine learning, this could involve validation tests to verify model accuracy and robustness under different scenarios.
    
-   **draw.py**: This Python script could be designed for generating visualizations, such as plots and charts, to illustrate the results of experiments or data analyses. Visualization is crucial for interpreting complex data and results in an accessible and meaningful way.
- -   **privacy.py and privacy-Instances.py**: These scripts are likely related to implementing privacy-preserving techniques or algorithms. The name suggests they contain functions or classes designed to enhance or ensure the privacy of data processed within your projects. The file with “副本” might be a copy or a variant of the original privacy.py script, potentially with some modifications or specific configurations.
    
-   **result_utils.py**: This script is probably used for handling and processing results generated by your machine learning models or experiments. It might include functions for summarizing results, performing statistical analysis, or generating reports that detail the performance and outcomes of your experiments.
    
-   **data_utils.py**: Typically, a data_utils script would contain utility functions related to data handling and manipulation, such as loading datasets, preprocessing data, and possibly performing transformations or augmentations needed for machine learning training.
    
-   **dlg.py**: Given the name, this script might be related to implementing a Deep Leakage from Gradients (DLG) attack or related functions. DLG is a type of attack that aims to reconstruct input data from gradients, thus posing a significant privacy risk in machine learning settings, especially in federated learning.
    
-   **mem_utils.py**: This file likely contains utilities related to memory management or data storage. In a machine learning context, this could involve optimizing memory usage during training, managing cache, or handling large datasets efficiently to prevent memory overflow.


##  Experimental Results
These graphs and tables summarize the effectiveness of different federated learning strategies under various attack vectors, focusing on how different privacy-preserving methods hold up against advanced adversarial techniques. Each plot and metric offers insight into the trade-offs between model accuracy, loss, and privacy protection.
![输入图片说明](/imgs/2024-06-16/ptRhoSn9YP22ESx2.png)
![输入图片说明](/imgs/2024-06-16/rYawiKXlj95mJril.png)
![输入图片说明](/imgs/2024-06-16/dRd4qx2JMUNTKYC3.png)



```
## Update log

```
- {24.06.13} Uploaded overall framework code and readme file
```

