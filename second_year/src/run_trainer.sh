mkdir ../results/
CUDA_VISIBLE_DEVICES=1 python train.py \
"model ./models/Causal_CRN_SPL_target/model.yaml" \
"dataloader ./dataloader/dataloader_train.yaml" \
"hyparam ./hyparam/train.yaml" \
"learner ./hyparam/learner.yaml" \
"logger ./hyparam/logger.yaml"\