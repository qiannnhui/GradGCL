#### GIN fine-tuning

#nohup ./finetune.sh bbbp > log_bbbp &
#nohup ./finetune.sh sider > log_sider &
#nohup ./finetune.sh toxcast > log_toxcast &

./finetune.sh bbbp 2
./finetune.sh sider 2
./finetune.sh toxcast 2