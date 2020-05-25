CHECKPOINT_ROOT=/home/jzda/storage/zhengant/VLP/checkpoints
DATA_ROOT=/home/jzda/storage/zhengant/fake_media_vlp

TMPDIR=/home/jzda/storage/zhengant/VLP/tmp/ \
PYTORCH_PRETRAINED_BERT_CACHE=/home/jzda/storage/zhengant/VLP/tmp/ \
PYTHONPATH=/home/jzda/storage/zhengant/VLP:$PYTHONPATH \
python vlp/decode_img2txt.py \
    --dataset fake_media \
    --model_recover_path $CHECKPOINT_ROOT/fake_media_ft_coco_cider/model.30.bin \
    --new_segment_ids --batch_size 100 --beam_size $2 --enable_butd \
    --image_root $DATA_ROOT/region_feat_gvd_wo_bgd --split $1 \
    --src_file $DATA_ROOT/dataset_vlp.json \
    --region_det_file_prefix feat_cls_1000/fake_media_detection_vg_100dets_vlp_checkpoint_trainval \
    --region_bbox_file raw_bbox/fake_media_detection_vg_100dets_vlp_checkpoint_trainval_bbox \
    --max_tgt_length 100 \
    --file_valid_jpgs "" 