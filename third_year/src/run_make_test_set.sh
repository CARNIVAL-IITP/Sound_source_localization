
device=0
echo "device: $device"

CUDA_VISIBLE_DEVICES=$device python make_test_set.py \
"model ./models/Causal_CRN_SPL_target/model.yaml" \
"dataloader ./dataloader/dataloader_test_maker.yaml" \
"hyparam ./hyparam/train.yaml" \
"learner ./hyparam/learner.yaml" \
"logger ./hyparam/logger.yaml"\