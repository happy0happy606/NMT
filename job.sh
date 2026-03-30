#!/bin/sh 

#home_dir='/SISDC_GPFS/Home_SE/suda-cst/ycli-suda/'
#data_dir=$home_dir'mt/exp/enfr/base/data/shards'
#model_dir=$home_dir'mt/exp/ijcai21/base60-enfr-gpu2/'

home_dir="/home/lijunhui/ycli/"
base_dir=$home_dir'g-transformer-main/'
data_dir=$base_dir'exp_sent/IWSLT2017.binarized.en-de/'
model_dir=$base_dir'exp_sent/'

data_set=$data_dir

mypython=python3
#mypython=$home_dir'/anaconda3/envs/pytorch13/bin/python'
#export device=0,1,2,3
export device=6
gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`
src_lan=en
tgt_lan=de
#BSUB -n 16
#BSUB -q HPC.S1.GPU.X785.sha
#BSUB -o %J.out
#BSUB -gpu "num=2:mode=exclusive_process"

CUDA_VISIBLE_DEVICES=$device \
$mypython       train.py $data_set \
                --save-dir $model_dir \
                --distributed-world-size $gpu_num \
                -s $src_lan -t $tgt_lan \
                --ddp-backend=no_c10d \
                --task translation \
                --arch transformer \
                --max-update 100000 \
                --save-interval-updates 0 \
                --keep-interval-updates 1000 \
                --share-all-embeddings \
                --optimizer adam \
                --adam-betas '(0.9, 0.98)' \
                --clip-norm 0.0 \
                --lr 5e-4 \
                --weight-decay 0.0001 \
                --lr-scheduler inverse_sqrt \
                --warmup-updates 4000 \
                --criterion label_smoothed_cross_entropy \
                --label-smoothing 0.1 \
                --max-tokens 4096 \
                --update-freq 2 \
                --log-interval 100 \