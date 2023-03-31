for env_file in "$@"
do
    echo -e "\n===========$env_file===========\n"
    sed -n '/Loading config from:/p' $env_file
    
    input_str="$(sed -n '/at https:\/\/wandb.ai\/etok\/etok\/runs/p' $env_file | grep -o 'https://wandb.ai/etok/etok/runs/........' | tail -n1)"
    str="${input_str:32}"
    find /scratch1/sghaneka/etok/etok/$str/checkpoints/ -name "*.ckpt"
    echo -e "\n=======================\n"
done