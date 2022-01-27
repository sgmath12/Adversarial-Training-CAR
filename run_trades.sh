#!/bin/bash



ALPHA_1=0.1
DIR="./check_point"
TRAIN_METHOD="TRADES"
LOSS="l2_loss"

python ./train.py --model_name resnet18_reg --batch_size 128 --alpha ${ALPHA_1} --training_method ${TRAIN_METHOD} --reg_loss ${LOSS} --best_model_path "${DIR}/resnet18_TRADES_${LOSS}_${ALPHA_1}_best.pth" --last_model_path "${DIR}/resnet18_TRADES_${LOSS}_${ALPHA_1}_last.pth"
