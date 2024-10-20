# Optimizing YOLOv8 for Lightweight and Accurate UAV-based Plant Detection





This repository contains the code supporting the research for the paper **"Optimizing YOLOv8 for Lightweight and Accurate UAV-based Plant Detection"** (currently under peer review). The modifications and experiments in this work are aimed at enhancing the YOLOv8 model to improve its performance for UAV-based plant detection tasks, balancing both accuracy and computational efficiency.



The repository includes:

- **`config`**: This folder contains the configuration files for the models used in the experiments described in the paper. Each file corresponds to a specific model setup and can be used to replicate the results.



In this study, we replaced the original CIoU loss function in YOLOv8 with the WIoU loss function to enhance the accuracy of bounding box regression for small plant targets. To implement WIoU in YOLOv8, follow the instructions below.

- Import `wiou_loss` (the `wiou_loss.py` file can be found in the `yolo` directory of this repository) into `ultralytics/utils/loss.py`.

- Modify the `forward` function in the `BboxLoss` class in `ultralytics/utils/loss.py` by adding the following code:
  
  ```python
  if config.bbox_loss == 'WIoU':
      loss, iou = bbox_wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
      loss_iou = (loss * weight).sum() / target_scores_sum
  else:   # CIoU
      iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
      loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
  ```