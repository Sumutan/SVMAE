# best for UCF

python main_test.py --dataset ucf --feat-extractor videomae --feature-size 1536 \
  --feature-group both --save_test_results --fusion concat --aggregate_text \
  --use_dic_gt --pretrained-model ./ckpt/a_test/ucf-8910.pkl \
  --aggregate_text --use_dic_gt \
  --rgb-list ./list/ucf-videoMae-CLIP-L_UCF_9-5_9-1_finetune_dif_0.5_SP_norm_a0.05_fast.list \
  --test-rgb-list ./list/ucf-videoMae-test-CLIP-L_UCF_9-5_9-1_finetune_dif_0.5_SP_norm_a0.05_fast.list \
  --exp-name 8910