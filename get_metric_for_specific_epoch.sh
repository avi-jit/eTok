for env_file in "$@"
do
    echo -e "\n===========$env_file===========\n"
    sed -n '/Loading config from:/p' $env_file
    echo -e "\n=======================\n"
    sed -n '/Training using following config:/,/- Epochs:/p' $env_file
    echo -e "\n=======================\n"
    
    big_str=$(grep -oE 'Epoch 58: 100%.{0,1000}' $env_file)
    if [[ -n $output ]]
    then
        echo 
    else
        echo "$(echo $big_str | grep -o 'Epoch...' | tail -n1)": "$(echo $big_str | grep -o 'acc_unit_epoch=.....' | tail -n1)"
        echo "$(echo $big_str | grep -o 'Epoch...' | tail -n1)": "$(echo $big_str | grep -o 'acc_word_epoch=.....' | tail -n1)"
        echo -e "\n=======================\n"
    fi
done
