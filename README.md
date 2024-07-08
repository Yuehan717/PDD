# [ECCV 2024] Pairwise Distance Distillation
### Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution  
[![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-<COLOR>.svg)](https://arxiv.org/abs/<INDEX>)    
Yuehan Zhang<sup>1</sup>, Seungjun Lee<sup>1,2</sup>, Angela Yao<sup>1</sup>  
National University of Singapore<sup>1</sup>, Korea University<sup>2</sup>  
<p align="center">
<img src="teaser.gif" width="800" />
</p>
  
## 📝PDD
We address the **unsupervised RWSR** for a targeted real-world degradation. We study from a distillation perspective and introduce a novel Pairwise Distance Distillation framework.
Through our framework, a model specialized in synthetic degradation adapts to target real-world degradations by distilling intra- and inter-model distances across the specialized model and an auxiliary generalized model. 

Our method, as a learning framework, can be applied to off-the-shelf generalist models, e.g., RealESRGAN, and improve their performance in a real-world domain!

### 🎯 TODOS
* Release model weights
* Release code
* Update visual comparisons
## 🖼️ Results

## 👓 Key Features
We tackle the unsupervised SR for a given real-world dataset through a distillation perspective: 
- we combine the knowledge from a Generalist (blind generalized model) and a Specialist (optimized for specific synthetic degradation);
- we perform the distillation for **distances** between features of predictions, rather than features themselves;
- only the Specialist is updated by gradients, and the full optimization consists of unsupervised distillation and supervised loss on synthetic data.

The distillation is based on the consistency of intra- and inter-model distances. We refer to the paper for explorations that establish these consistencies. 😃

We provide an EMA configuration that only uses a single pretrained model. Specifically, the Generalist and Specialist are initialized by the same blind-generalized model, and the Generalist is kept as an EMA version of the Specialist during distillation.  
Experimentally, the EMA configuration achieves the best performance.

## 🔨 Installation
Instructions on how to set up the environment and dependencies required to run the code. Provide step-by-step commands:
```sh
# Clone the repository
git clone https://github.com/Yuehan717/PDD.git

# Navigate into the repository
cd PDD

# Install dependencies
pip install -r requirements.txt
```
## 👉 Datasets & Model Weights
#### Datasets
Our method requires two sets of data for training:
- Paired synthetic data: We use ground-truth images in [DF2K]() and create LRs on the fly;
- LR real-world data: In the paper, we experiment with RealSR, DRealSR, NTRIE20; the efficiency of our method can also be validated on dped and video dataset VideoLQ.

#### Model Weights
We provide model weights of RealESRAGN and BSRGAN trained with our method.
| Domain | Google Drive | Baidu Pan |
|----------|----------|----------|
| Row 1    | Cell 2   | Cell 3   |
| Row 2    | Cell 5   | Cell 6   |
| Row 3    | Cell 8   | Cell 9   |

## 👉 Usage
```sh
# Testing command

# Training command
```
We use [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) for evaluation. Thanks to their great work👏

## 🔎 Limitations
Like all other works, our method is not perfect. We have found limitations that:
- The method is more effective in improving blurry reconstructions than other effects.
- The ratio between intra- and inter-distance distillation terms will result in a perception-distortion trade-off of the final results. The ratio now relies on hyperparameter tuning.


## 👏 Acknowledgement
The code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks to their great contribution to the area!
