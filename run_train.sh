CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
                        -b 108 \
                        --epochs 250 \
                        --lr 0.0001 \
                        --arch VGG \
                        --snapshot-fname-prefix exps/snapshots/pil_lip_vgg \
                        --pred-path exps/preds/csv_results/pred_keypoints_lip_vgg.csv \
                        2>&1 | tee exps/logs/pil_lip_vgg.log \
