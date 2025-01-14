#!/bin/bash

deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 /home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/code/train.py\
    --model nextgpt \
    --stage 2\
    --save_path  /home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/test\
    --log_path /home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/test\
