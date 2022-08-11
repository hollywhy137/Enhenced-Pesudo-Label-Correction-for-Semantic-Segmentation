# Enhanced Pseudo Label Correction for Domain Adaptive Semantic Segmentation

## Master's of Data Science Research Thesis code documentation


### Summary 
This Study focuses on the unsupervised domain adaptive semantic segmentation tasks that has drawn numerous people’s attention in recent years. A range of outstand-ing literature recently documents that the combination of pseudo learning and domain adaptation in semantic segmentation can generate state-of-art performance. The use of pseudo labels transform the original unsupervised learning problem to a semi-supervised learning problem and convey the idea that many well-studied semi-supervised learning techniques can be applied to adaptive semantic segmentation tasks. Inspired by some latest papers that utilise pseudo learning techniques in image classification tasks, we are motivated to explore whether MC-dropout and pseudo label selection strategies would lead to better performance in the semantic segmentation context. Thus, we propose a pseudo label correction method aiming to improve the quality of pseudo labels in the training stage and hence boost the final performance. We implement the method upon the state-of-art model, [MRNet](https://github.com/layumi/Seg-Uncertainty). Consequently, we evaluate the effectiveness of the pro- posed method using two widely used synthetic-to-real benchmarks as well as a cross-city benchmark. Through extensive experiments, we illustrate that the proposed method can achieve significant improvement of prediction accuracy in all experiments compared with various competitive benchmarks using different approaches.

### Prerequisites
- Python 3.6
- GPU Memory >= 11G (e.g., GTX2080Ti or GTX1080Ti)
- Pytorch 

### Training
Step 1:
``` 
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0,1  --often-balance  --use-se 
```
Step 2:
``` 
python generate_plabel_cityscapes.py --restore-from ./snapshots/GTA_stage1/GTA5_25000.pth
```
Step 3:
``` 
python train_pn_plabel_GTA.py \
    --snapshot-dir ./transition/GTA/snapshots/strategy_2\
    --restore-from ./paper_snapshot/GTA_stage1/GTA5_25000.pth\
    --drop 0.2 \
    --warm-up 5000 \
    --batch-size 9 \
    --learning-rate 1e-4 \
    --crop-size 512,256 \
    --lambda-seg 0.5 \
    --lambda-adv-target1 0 \
    --lambda-adv-target2 0 \
    --lambda-me-target 0 \
    --lambda-kl-target 0 \
    --norm-style gn \
    --class-balance \
    --only-hard-label 80 \
    --max-value 7 \
    --gpu-ids 0,1,2 \
    --often-balance  \
    --use-se  \
    --input-size 1280,640  \
    --train_bn  \
    --autoaug False
```
### Testing
```
python evaluate_cityscapes.py --restore-from ./snapshots/GTA_stage2_0613/GTA5_100000.pth
```
