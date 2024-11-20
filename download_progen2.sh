#!/bin/bash

for model in progen2-small progen2-medium progen2-base progen2-large
do
    wget -P ./progen2-checkpoints/${model} https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz
    tar -xvf ./progen2-checkpoints/${model}/${model}.tar.gz -C ./progen2-checkpoints/${model}/
done