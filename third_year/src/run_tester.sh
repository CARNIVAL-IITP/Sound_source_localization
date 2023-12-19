

device=0
echo "device: $device"

CUDA_VISIBLE_DEVICES=$device python inference.py \
"model ./models/Causal_CRN_SPL_target/model.yaml" \
"dataloader ./dataloader/dataloader_test.yaml" \
"hyparam ./hyparam/test.yaml" \
"learner ./hyparam/learner.yaml" \
"logger ./hyparam/logger.yaml"\