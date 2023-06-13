for env_file in "$@"
do
    echo -e "\n===========$env_file===========\n"
    sed -n '/Loading config from:/p' $env_file
    echo -e "\n=======================\n"
    sed -n '/Training using following config:/,/- Epochs:/p' $env_file
    echo -e "\n=======================\n"
    sed -n '/| Name/,/Total estimated model params size (MB)/p' $env_file
    echo -e "\n=======================\n"
    sed -n '/data has/,/self.maxlen/p' $env_file
    echo -e "\n=======================\n"

    output=$(sed -n '/wandb: Run summary:/,/wandb: trainer/p' $env_file)
    if [[ -n $output ]]
    then
        echo "$output"
    else
        echo wandb: "$(grep -o 'acc_unit_epoch=.....' $env_file | tail -n1)"
        echo wandb: "$(grep -o 'acc_word_epoch=.....' $env_file | tail -n1)"
        echo wandb: "$(grep -o 'gpu_epoch=.....' $env_file | tail -n1)"
        echo wandb: killed at "$(grep -o 'Epoch...' $env_file | tail -n1)"
    fi

    sed -n '/at https:\/\/wandb.ai\/etok\/etok\/runs/p' $env_file

    input_str="$(sed -n '/at https:\/\/wandb.ai\/etok\/etok\/runs/p' $env_file | grep -o 'https://wandb.ai/etok/etok/runs/........' | tail -n1)"
    str="${input_str:32}"
    echo model_checkpoint_path: "$(find /scratch1/sghaneka/etok/etok/$str/checkpoints/ -name '*.ckpt')"
    echo -e "\n=======================\n"
done
