device=0


CUDA_VISIBLE_DEVICES=$device python inference.py \
"model ./models/Causal_CRN_SPL_target/model.yaml" \
"dataloader ./dataloader/dataloader_test.yaml" \
"hyparam ./hyparam/test.yaml" \
