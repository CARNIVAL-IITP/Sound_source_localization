

CUDA_VISIBLE_DEVICES=0 python srp_phat_inference.py \
"model ./models/Causal_CRN_SPL_target/model.yaml" \
"dataloader ./dataloader/dataloader_test.yaml" \
"hyparam ./hyparam/test.yaml" \
"learner ./hyparam/learner.yaml" \
"logger ./hyparam/logger.yaml"\