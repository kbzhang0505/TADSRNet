# TADSRNet
After the paper is accepted, we will provide code for the paper: TADSRNet: A triple-attention dual-scale residual network for super-resolution image quality assessment.
# Abstract
Image super-resolution (SR) has been extensively investigated in recent years. However, due to the absence of trustworthy and precise perceptual quality standards, it is challenging to objectively measure the performance of different SR approaches. In this paper, we propose a novel triple attention dual-scale residual network called TADSRNet for no-reference super-resolution image quality assessment (NR-SRIQA). Firstly, we simulate the human visual system (HVS) and construct a triple attention mechanism to acquire more significant portions of SR images through cross-dimensionality, making it simpler to identify visually sensitive regions. Then a dual-scale convolution module (DSCM) is constructed to capture quality-perceived features at different scales. Furthermore, in order to collect more informative feature representation, a residual connection is added to the network to compensate for perceptual features. Extensive experimental results demonstrate that the proposed TADSRNet can predict visual quality with greater accuracy and better consistency with human perception compared with existing IQA methods. The code will be available at https://github.com/kbzhang0505/TADSRNet. 
# The overall framework
https://github.com/kbzhang0505/TADSRNet/blob/main/Fig%202.png
# Quantitative results
https://github.com/kbzhang0505/TADSRNet/blob/main/Fig%201.png
# Train
`python main.py`

# Citation
If you find the code helpful in your research or work, please cite the following paper:
    @article{quan2023tadsrnet,
       title={TADSRNet: A triple-attention dual-scale residual network for super-resolution image quality assessment},
       author={Quan, Xing and Zhang, Kaibing and Li, Hui and Fan, Dandan and Hu, Yanting and Chen, Jinguang},
       journal={Applied Intelligence},
       volume={53},
       number={22},
       pages={26708--26724},
       year={2023},
       publisher={Springer}
    }    
With any questions, feel welcome to contact us!
