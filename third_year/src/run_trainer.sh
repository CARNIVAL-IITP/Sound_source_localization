
device=0
echo "device: $device"

CUDA_VISIBLE_DEVICES=$device python train.py \
"model ./models/Causal_CRN_SPL_target/model.yaml" \
"dataloader ./dataloader/dataloader_train.yaml" \
"hyparam ./hyparam/train.yaml" \
"learner ./hyparam/learner.yaml" \
"logger ./hyparam/logger.yaml"\

