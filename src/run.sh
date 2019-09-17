TAB='single'
GPU_IDS=0
DATASET='h36m_up_mpii_aich_pa'

TEST_MODE=0
EVAL_MODE=1

VBS=16
EVAL_protocol='1'
EVAL_DATASET='h36m'

if [ "$EVAL_MODE" = 1 ]
then
    #CUDA_VISIBLE_DEVICES=${GPU_IDS} python3 test.py --gpu=${GPU_IDS} --tab=${TAB} --dataset=${EVAL_DATASET} --val_batch_size=${VBS} --eval --eval-pw3d --video --eval-with-single-frame-network=1 > ${TAB}'_'${EVAL_DATASET}'_g'${GPU_IDS}.log 2>&1 &
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python3 test.py --gpu=${GPU_IDS} --tab=${TAB} --dataset=${EVAL_DATASET} --val_batch_size=${VBS} --eval --test-single
    #CUDA_VISIBLE_DEVICES=${GPU_IDS} python3 test.py --gpu=${GPU_IDS} --tab=${TAB} --dataset=${EVAL_DATASET} --val_batch_size=${VBS} --eval --eval-protocol=${EVAL_protocol}
elif [ "$TEST_MODE" = 1 ]
then
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python3 train.py --gpu=${GPU_IDS} --tab=${TAB} --dataset=${DATASET}
else
    CUDA_VISIBLE_DEVICES=${GPU_IDS} nohup python3 train.py --gpu=${GPU_IDS} --tab=${TAB} --dataset=${DATASET} > ${TAB}'_'${DATASET}'_g'${GPU_IDS}.log 2>&1 &
fi

# arguments list
#_--with-kps --fine-tune --eval-pw3d --eval --save-features --video --eval-with-single-frame-network --save-obj --save-smpl-params --visual-all