# make tempfiles here instead of /tmp to avoid running out of space

CHECKPOINT_ROOT=/home/jzda/storage/zhengant/VLP/checkpoints
DATA_ROOT=/home/jzda/storage/zhengant/fake_media_vlp

# set tmpdir and dir for pretrained bert to avoid putting stuff on the full drive
TMPDIR=/home/jzda/storage/zhengant/VLP/tmp/ \
PYTORCH_PRETRAINED_BERT_CACHE=/home/jzda/storage/zhengant/tmp/.pytorch_pretrained_bert/ \
python vlp/run_img2txt_dist.py --output_dir $CHECKPOINT_ROOT/fake_media_ft_coco_cider_scst \
    --model_recover_path $CHECKPOINT_ROOT/coco_g4_lr1e-6_batch64_scst/model.19.bin \
    --do_train --new_segment_ids --always_truncate_tail --amp \
    --src_file $DATA_ROOT/dataset_vlp.json \
    --file_valid_jpgs /dev/null \
    --image_root $DATA_ROOT/region_feat_gvd_wo_bgd \
    --region_det_file_prefix feat_cls_1000/fake_media_detection_vg_100dets_vlp_checkpoint_trainval \
    --region_bbox_file raw_bbox/fake_media_detection_vg_100dets_vlp_checkpoint_trainval_bbox \
    --enable_butd --s2s_prob 1 --bi_prob 0 \
    --max_pred 0 --mask_prob 0 --scst