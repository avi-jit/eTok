# Report

## Files that are successfully completed:

### Dataset: custom_en

- configs/custom_en_byte_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/fwp94zr0
  - slurm-14187613.out

- configs/custom_en_char_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/lkmq8z0e
  - slurm-14187615.out

- configs/custom_en_sub_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/zarb3lic
  - slurm-14187616.out

- configs/custom_en_word_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/6ym5d28h
  - slurm-14187612.out

- configs/custom_en_byte_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/kxc5tfyj
  - slurm-14187609.out

- configs/custom_en_char_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/tshrru6g
  - slurm-14187610.out

### Dataset: custom_fr

- configs/custom_fr_char_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/phfttptc
  - slurm-14187622.out

- configs/custom_fr_word_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/pbifulwi
  - slurm-14187620.out

- configs/custom_fr_byte_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/neslbjdm
  - slurm-14187621.out

- configs/custom_fr_sub_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/nk70pi95
  - slurm-14187623.out

- configs/custom_fr_char_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/n76bmh8w
  - slurm-14187618.out

- configs/custom_fr_sub_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/czuqq74c
  - slurm-14187619.out

### Dataset: custom_ru

- configs/custom_ru_byte_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/at2fy9mr
  - slurm-14187606.out

- configs/custom_ru_char_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/3xum2c7j
  - slurm-14187607.out

- configs/custom_ru_sub_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/afkj9f4n
  - slurm-14187608.out

- configs/custom_ru_word_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/znw2h2on
  - slurm-14187605.out

- configs/custom_ru_char_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/6rych95j
  - slurm-14187602.out

- configs/custom_ru_sub_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/baf4jwnm
  - slurm-14187604.out

- configs/custom_ru_byte_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/i1q3oiwq
  - slurm-14187601.out

### Dataset: shakespeare_en

- configs/shakespeare_en_char_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/u107s5sz
  - slurm-14187629.out

- configs/shakespeare_en_sub_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/due0lggq
  - slurm-14187630.out

- configs/shakespeare_en_byte_e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/4xafmof9
  - slurm-14187628.out

- configs/shakespeare_en_word_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/alkgjqf6
  - slurm-14187627.out

- configs/shakespeare_en_sub_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/7y6fn93z
  - slurm-14187626.out

- configs/shakespeare_en_char_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/3uatax1k
  - slurm-14187625.out

- configs/shakespeare_en_byte_no-e2e_0.0001_4_2.env

  - https://wandb.ai/etok/etok/runs/t45gj9nx
  - slurm-14187624.out

## Files causing errors:

- configs/custom_fr_byte_no-e2e_0.0001_4_2.env

  - ```
    File "/home1/sghaneka/eTok/newmodel.py", line 231, in validation_step
        inputs[i] = row[last_word_beg - context+1:last_word_beg+1] # +1 to include space
    RuntimeError: The expanded size of the tensor (90) must match the existing size (0) at non-singleton dimension 0.  Target sizes: [90].  Tensor sizes: [0]
    ```

  - slurm-14187617.out

- configs/custom_en_sub_no-e2e_0.0001_4_2.env

  - ```
        [Previous line repeated 3 more times]
    File "/home1/sghaneka/eTok/dataset.py", line 321, in __getitem__
        raise NotImplemented
    ```

  - slurm-14187611.out

sh get_metric.sh "slurm_logs/result_2_logs/slurm-14187609.out" "slurm_logs/result_2_logs/slurm-14187610.out" "slurm_logs/result_2_logs/slurm-14187612.out" "slurm_logs/result_2_logs/slurm-14187613.out" "slurm_logs/result_2_logs/slurm-14187615.out" "slurm_logs/result_2_logs/slurm-14187616.out" > base_metric_custom_en.txt
sh get_metric.sh "slurm_logs/result_2_logs/slurm-14187618.out" "slurm_logs/result_2_logs/slurm-14187619.out" "slurm_logs/result_2_logs/slurm-14187620.out" "slurm_logs/result_2_logs/slurm-14187621.out" "slurm_logs/result_2_logs/slurm-14187622.out" "slurm_logs/result_2_logs/slurm-14187623.out" > base_metric_custom_fr.txt
sh get_metric.sh "slurm_logs/result_2_logs/slurm-14187601.out" "slurm_logs/result_2_logs/slurm-14187602.out" "slurm_logs/result_2_logs/slurm-14187604.out" "slurm_logs/result_2_logs/slurm-14187605.out" "slurm_logs/result_2_logs/slurm-14187606.out" "slurm_logs/result_2_logs/slurm-14187607.out" "slurm_logs/result_2_logs/slurm-14187608.out" > base_metric_custom_ru.txt
sh get_metric.sh "slurm_logs/result_2_logs/slurm-14187624.out" "slurm_logs/result_2_logs/slurm-14187625.out" "slurm_logs/result_2_logs/slurm-14187626.out" "slurm_logs/result_2_logs/slurm-14187627.out" "slurm_logs/result_2_logs/slurm-14187628.out" "slurm_logs/result_2_logs/slurm-14187629.out" "slurm_logs/result_2_logs/slurm-14187630.out" > base_metric_shakespeare_en.txt

sh get_checkout_paths.sh "slurm_logs/result_2_logs/slurm-14187609.out" "slurm_logs/result_2_logs/slurm-14187610.out" "slurm_logs/result_2_logs/slurm-14187612.out" "slurm_logs/result_2_logs/slurm-14187613.out" "slurm_logs/result_2_logs/slurm-14187615.out" "slurm_logs/result_2_logs/slurm-14187616.out" "slurm_logs/result_2_logs/slurm-14187618.out" "slurm_logs/result_2_logs/slurm-14187619.out" "slurm_logs/result_2_logs/slurm-14187620.out" "slurm_logs/result_2_logs/slurm-14187621.out" "slurm_logs/result_2_logs/slurm-14187622.out" "slurm_logs/result_2_logs/slurm-14187623.out" "slurm_logs/result_2_logs/slurm-14187601.out" "slurm_logs/result_2_logs/slurm-14187602.out" "slurm_logs/result_2_logs/slurm-14187604.out" "slurm_logs/result_2_logs/slurm-14187605.out" "slurm_logs/result_2_logs/slurm-14187606.out" "slurm_logs/result_2_logs/slurm-14187607.out" "slurm_logs/result_2_logs/slurm-14187608.out" "slurm_logs/result_2_logs/slurm-14187624.out" "slurm_logs/result_2_logs/slurm-14187625.out" "slurm_logs/result_2_logs/slurm-14187626.out" "slurm_logs/result_2_logs/slurm-14187627.out" "slurm_logs/result_2_logs/slurm-14187628.out" "slurm_logs/result_2_logs/slurm-14187629.out" "slurm_logs/result_2_logs/slurm-14187630.out" > checkpoint_paths.txt

sh train_config.sh "configs/custom_ru_byte_no-e2e_0.0001_4_2.env" "configs/custom_ru_char_no-e2e_0.0001_4_2.env" "configs/custom_ru_sub_no-e2e_0.0001_4_2.env" "configs/custom_ru_word_no-e2e_0.0001_4_2.env" "configs/custom_ru_byte_e2e_0.0001_4_2.env" "configs/custom_ru_char_e2e_0.0001_4_2.env" "configs/custom_ru_sub_e2e_0.0001_4_2.env" "configs/custom_en_byte_no-e2e_0.0001_4_2.env" "configs/custom_en_char_no-e2e_0.0001_4_2.env" "configs/custom_en_sub_no-e2e_0.0001_4_2.env" "configs/custom_en_word_no-e2e_0.0001_4_2.env" "configs/custom_en_byte_e2e_0.0001_4_2.env" "configs/custom_en_char_e2e_0.0001_4_2.env" "configs/custom_en_sub_e2e_0.0001_4_2.env" "configs/custom_fr_byte_no-e2e_0.0001_4_2.env" "configs/custom_fr_char_no-e2e_0.0001_4_2.env" "configs/custom_fr_sub_no-e2e_0.0001_4_2.env" "configs/custom_fr_word_no-e2e_0.0001_4_2.env" "configs/custom_fr_byte_e2e_0.0001_4_2.env" "configs/custom_fr_char_e2e_0.0001_4_2.env" "configs/custom_fr_sub_e2e_0.0001_4_2.env" "configs/shakespeare_en_byte_no-e2e_0.0001_4_2.env" "configs/shakespeare_en_char_no-e2e_0.0001_4_2.env" "configs/shakespeare_en_sub_no-e2e_0.0001_4_2.env" "configs/shakespeare_en_word_no-e2e_0.0001_4_2.env" "configs/shakespeare_en_byte_e2e_0.0001_4_2.env" "configs/shakespeare_en_char_e2e_0.0001_4_2.env" "configs/shakespeare_en_sub_e2e_0.0001_4_2.env"

sh validation_config.sh "configs/val_custom_ru_byte_no-e2e_0.0001_4_2.env" "configs/val_custom_ru_char_no-e2e_0.0001_4_2.env" "configs/val_custom_ru_sub_no-e2e_0.0001_4_2.env" "configs/val_custom_ru_word_no-e2e_0.0001_4_2.env" "configs/val_custom_ru_byte_e2e_0.0001_4_2.env" "configs/val_custom_ru_char_e2e_0.0001_4_2.env" "configs/val_custom_ru_sub_e2e_0.0001_4_2.env" "configs/val_custom_en_byte_no-e2e_0.0001_4_2.env" "configs/val_custom_en_char_no-e2e_0.0001_4_2.env" "configs/val_custom_en_word_no-e2e_0.0001_4_2.env" "configs/val_custom_en_byte_e2e_0.0001_4_2.env" "configs/val_custom_en_char_e2e_0.0001_4_2.env" "configs/val_custom_en_sub_e2e_0.0001_4_2.env" "configs/val_custom_fr_char_no-e2e_0.0001_4_2.env" "configs/val_custom_fr_sub_no-e2e_0.0001_4_2.env" "configs/val_custom_fr_word_no-e2e_0.0001_4_2.env" "configs/val_custom_fr_byte_e2e_0.0001_4_2.env" "configs/val_custom_fr_char_e2e_0.0001_4_2.env" "configs/val_custom_fr_sub_e2e_0.0001_4_2.env" "configs/val_shakespeare_en_byte_no-e2e_0.0001_4_2.env" "configs/val_shakespeare_en_char_no-e2e_0.0001_4_2.env" "configs/val_shakespeare_en_sub_no-e2e_0.0001_4_2.env" "configs/val_shakespeare_en_word_no-e2e_0.0001_4_2.env" "configs/val_shakespeare_en_byte_e2e_0.0001_4_2.env" "configs/val_shakespeare_en_char_e2e_0.0001_4_2.env" "configs/val_shakespeare_en_sub_e2e_0.0001_4_2.env"

issue in slurm-14213961.out
slurm-14254978.out : same input error in this file too
issue in configs/val_custom_ru_byte_no-e2e_0.0001_4_2.env
slurm-14213957.out ru-sub-false

val_error: slurm-14265845.out

train: "slurm-14254979.out" "slurm-14254978.out"

sh get_metric.sh "slurm-14213956.out" "slurm-14213958.out" "slurm-14213959.out" "slurm-14213960.out" "slurm-14213962.out" "slurm-14213963.out" "slurm-14213955.out" "slurm-14213965.out" > base_metric_mix_1.txt

sh validation_config.sh "configs/val_custom_ru_byte_no-e2e_0.0001_4_2.env" "configs/val_custom_fr_sub_e2e_0.0001_4_2.env"

sh validation_config.sh "configs/val_custom_ru_char_no-e2e_0.0001_4_2.env" "configs/val_custom_ru_sub_e2e_0.0001_4_2.env" "configs/val_custom_en_byte_no-e2e_0.0001_4_2.env" "configs/val_custom_en_char_no-e2e_0.0001_4_2.env" "configs/val_custom_en_sub_e2e_0.0001_4_2.env" "configs/val_custom_fr_char_no-e2e_0.0001_4_2.env"

python eval_script.py --val_output_file_paths val_output_custom_en_byte_e2e_0.0001_4_8.csv val_output_custom_en_byte_no-e2e_0.0001_0_8.csv val_output_custom_en_char_e2e_0.0001_4_8.csv val_output_custom_en_char_no-e2e_0.0001_0_8.csv val_output_custom_en_sub_e2e_0.0001_4_8.csv val_output_custom_en_word_no-e2e_0.0001_0_8.csv val_output_custom_fr_byte_e2e_0.0001_4_8.csv val_output_custom_fr_char_e2e_0.0001_4_8.csv val_output_custom_fr_char_no-e2e_0.0001_0_8.csv val_output_custom_fr_sub_e2e_0.0001_4_8.csv val_output_custom_fr_sub_no-e2e_0.0001_0_8.csv val_output_custom_fr_word_no-e2e_0.0001_0_8.csv

python eval_script.py --val_output_file_paths val_output_custom_ru_byte_no-e2e_0.0001_0_8.csv

python eval_script.py --val_output_file_paths val_output_custom_ru_byte_no-e2e_0.0001_4_8.csv val_output_custom_fr_sub_e2e_0.0001_4_8.csv --output_file eval_results_updated.json

python eval_script.py --val_output_file_paths val_output_custom_ru_byte_e2e_0.0001_4_8.csv val_output_custom_ru_char_e2e_0.0001_4_8.csv val_output_custom_ru_char_no-e2e_0.0001_0_8.csv val_output_custom_ru_sub_e2e_0.0001_4_8.csv val_output_custom_ru_sub_no-e2e_0.0001_0_8.csv val_output_custom_ru_word_no-e2e_0.0001_0_8.csv val_output_shakespeare_en_byte_e2e_0.0001_4_8.csv val_output_shakespeare_en_byte_no-e2e_0.0001_0_8.csv val_output_shakespeare_en_char_e2e_0.0001_4_8.csv val_output_shakespeare_en_char_no-e2e_0.0001_0_8.csv val_output_shakespeare_en_sub_e2e_0.0001_4_8.csv val_output_shakespeare_en_sub_no-e2e_0.0001_0_8.csv val_output_shakespeare_en_word_no-e2e_0.0001_0_8.csv

python eval_script.py --val_output_file_paths "val_output_custom_en_byte_no-e2e_0.0001_0_8.csv" "val_output_custom_en_char_no-e2e_0.0001_0_8.csv" "val_output_custom_en_sub_e2e_0.0001_4_8.csv" "val_output_custom_fr_char_no-e2e_0.0001_0_8.csv" "val_output_custom_ru_char_no-e2e_0.0001_0_8.csv" "val_output_custom_ru_sub_e2e_0.0001_4_8.csv" --output_file eval_results_updated.json

sh get_metric_for_specific_epoch.sh "slurm_logs/result_2_logs/slurm-14187609.out" "slurm_logs/result_2_logs/slurm-14187610.out" "slurm_logs/result_2_logs/slurm-14187612.out" "slurm_logs/result_2_logs/slurm-14187613.out" "slurm_logs/result_2_logs/slurm-14187615.out" "slurm_logs/result_2_logs/slurm-14187616.out" > base_metric_epoch_58_custom_en.txt && sh get_metric_for_specific_epoch.sh "slurm_logs/result_2_logs/slurm-14187618.out" "slurm_logs/result_2_logs/slurm-14187619.out" "slurm_logs/result_2_logs/slurm-14187620.out" "slurm_logs/result_2_logs/slurm-14187621.out" "slurm_logs/result_2_logs/slurm-14187622.out" "slurm_logs/result_2_logs/slurm-14187623.out" > base_metric_epoch_58_custom_fr.txt && sh get_metric_for_specific_epoch.sh "slurm_logs/result_2_logs/slurm-14187601.out" "slurm_logs/result_2_logs/slurm-14187602.out" "slurm_logs/result_2_logs/slurm-14187604.out" "slurm_logs/result_2_logs/slurm-14187605.out" "slurm_logs/result_2_logs/slurm-14187606.out" "slurm_logs/result_2_logs/slurm-14187607.out" "slurm_logs/result_2_logs/slurm-14187608.out" > base_metric_epoch_58_custom_ru.txt && sh get_metric_for_specific_epoch.sh "slurm_logs/result_2_logs/slurm-14187624.out" "slurm_logs/result_2_logs/slurm-14187625.out" "slurm_logs/result_2_logs/slurm-14187626.out" "slurm_logs/result_2_logs/slurm-14187627.out" "slurm_logs/result_2_logs/slurm-14187628.out" "slurm_logs/result_2_logs/slurm-14187629.out" "slurm_logs/result_2_logs/slurm-14187630.out" > base_metric_epoch_58_shakespeare_en.txt

sh get_metric_for_specific_epoch.sh "slurm-14213956.out" "slurm-14213958.out" "slurm-14213959.out" "slurm-14213960.out" "slurm-14213962.out" "slurm-14213963.out" "slurm-14213955.out" "slurm-14213965.out" "slurm-14213957.out" > base_metric_epoch_58_mix.txt

sh get_metric.sh "slurm-14254979.out" "slurm-14254978.out"

sh validation_config.sh "configs/val_custom_ru_sub_no-e2e_0.0001_4_2.env" "configs/val_custom_fr_byte_e2e_0.0001_4_2.env" "configs/val_custom_fr_sub_no-e2e_0.0001_4_2.env"

sh train_config.sh "configs/custom_ru_byte_e2e_0.0001_1_2.env" "configs/custom_ru_char_e2e_0.0001_1_2.env" "configs/custom_ru_sub_e2e_0.0001_1_2.env" "configs/custom_en_byte_e2e_0.0001_1_2.env" "configs/custom_en_char_e2e_0.0001_1_2.env" "configs/custom_en_sub_e2e_0.0001_1_2.env" "configs/custom_fr_byte_e2e_0.0001_1_2.env" "configs/custom_fr_char_e2e_0.0001_1_2.env" "configs/custom_fr_sub_e2e_0.0001_1_2.env" "configs/shakespeare_en_byte_e2e_0.0001_1_2.env" "configs/shakespeare_en_char_e2e_0.0001_1_2.env" "configs/shakespeare_en_sub_e2e_0.0001_1_2.env"

sh get_metric.sh "slurm-14313835.out" "slurm-14313836.out" "slurm-14313837.out" "slurm-14313838.out" "slurm-14313839.out" "slurm-14313840.out" "slurm-14313841.out" "slurm-14313842.out" "slurm-14313843.out" "slurm-14313844.out" "slurm-14313845.out" "slurm-14313846.out"

sh validation_config.sh "configs/val_custom_en_byte_e2e_0.0001_1_2.env" "configs/val_custom_en_char_e2e_0.0001_1_2.env" "configs/val_custom_en_sub_e2e_0.0001_1_2.env" "configs/val_custom_ru_byte_e2e_0.0001_1_2.env" "configs/val_custom_ru_char_e2e_0.0001_1_2.env" "configs/val_custom_ru_sub_e2e_0.0001_1_2.env" "configs/val_custom_fr_byte_e2e_0.0001_1_2.env" "configs/val_custom_fr_char_e2e_0.0001_1_2.env" "configs/val_custom_fr_sub_e2e_0.0001_1_2.env" "configs/val_shakespeare_en_byte_e2e_0.0001_1_2.env" "configs/val_shakespeare_en_char_e2e_0.0001_1_2.env" "configs/val_shakespeare_en_sub_e2e_0.0001_1_2.env"

python eval_script.py --val_output_file_paths "val_output_custom_en_byte_e2e_0.0001_1_8.csv" "val_output_custom_en_char_e2e_0.0001_1_8.csv" "val_output_custom_en_sub_e2e_0.0001_1_8.csv" "val_output_custom_fr_byte_e2e_0.0001_1_8.csv" "val_output_custom_fr_char_e2e_0.0001_1_8.csv" "val_output_custom_fr_sub_e2e_0.0001_1_8.csv" "val_output_custom_ru_byte_e2e_0.0001_1_8.csv" "val_output_custom_ru_char_e2e_0.0001_1_8.csv" "val_output_custom_ru_sub_e2e_0.0001_1_8.csv" "val_output_shakespeare_en_byte_e2e_0.0001_1_8.csv" "val_output_shakespeare_en_char_e2e_0.0001_1_8.csv" "val_output_shakespeare_en_sub_e2e_0.0001_1_8.csv" --output_file eval_results_1_prefix.json

sh train_config.sh "configs/shakespeare_en_byte_no-e2e_0.0001_4_2.env" "configs/shakespeare_en_char_no-e2e_0.0001_4_2.env" "configs/shakespeare_en_sub_no-e2e_0.0001_4_2.env" "configs/shakespeare_en_word_no-e2e_0.0001_4_2.env" "configs/shakespeare_en_byte_e2e_0.0001_4_2.env" "configs/shakespeare_en_char_e2e_0.0001_4_2.env" "configs/shakespeare_en_sub_e2e_0.0001_4_2.env"

sh train_config.sh "configs/custom_ru_byte_no-e2e_0.0001_4_2.env" "configs/custom_ru_char_no-e2e_0.0001_4_2.env" "configs/custom_ru_sub_no-e2e_0.0001_4_2.env" "configs/custom_ru_word_no-e2e_0.0001_4_2.env" "configs/custom_ru_byte_e2e_0.0001_4_2.env" "configs/custom_ru_char_e2e_0.0001_4_2.env" "configs/custom_ru_sub_e2e_0.0001_4_2.env" "configs/custom_en_byte_no-e2e_0.0001_4_2.env" "configs/custom_en_char_no-e2e_0.0001_4_2.env" "configs/custom_en_sub_no-e2e_0.0001_4_2.env" "configs/custom_en_word_no-e2e_0.0001_4_2.env" "configs/custom_en_byte_e2e_0.0001_4_2.env" "configs/custom_en_char_e2e_0.0001_4_2.env" "configs/custom_en_sub_e2e_0.0001_4_2.env" "configs/custom_fr_byte_no-e2e_0.0001_4_2.env" "configs/custom_fr_char_no-e2e_0.0001_4_2.env" "configs/custom_fr_sub_no-e2e_0.0001_4_2.env" "configs/custom_fr_word_no-e2e_0.0001_4_2.env" "configs/custom_fr_byte_e2e_0.0001_4_2.env" "configs/custom_fr_char_e2e_0.0001_4_2.env" "configs/custom_fr_sub_e2e_0.0001_4_2.env"

Failed due to sad errors:
{
{
"slurm-14545386.out": "custom_en_sub_no-e2e",
"error": "File '/home1/sghaneka/eTok/dataset.py', line 323, in **getitem**",
},
}

sh get_metric.sh "slurm-14542991.out" "slurm-14542992.out" "slurm-14542993.out" "slurm-14542994.out" "slurm-14542995.out" "slurm-14542996.out" "slurm-14542997.out" "slurm-14544865.out" "slurm-14544930.out" "slurm-14544995.out" "slurm-14545061.out" "slurm-14545125.out" > some_metric.txt

slurm_logs/result_2_logs/slurm-14213955.out
slurm_logs/result_2_logs/slurm-14213956.out
slurm_logs/result_2_logs/slurm-14213957.out

sh get_metric.sh "slurm-15043127.out" "slurm-15043128.out" "slurm-15043129.out"

sh train_config.sh "configs/wiki-convert_en_byte_no-e2e_true_0.0001_4_2.env" "configs/wiki-convert_en_char_no-e2e_true_0.0001_4_2.env" "configs/wiki-convert_en_sub_no-e2e_true_0.0001_4_2.env" "configs/wiki-convert_en_word_no-e2e_true_0.0001_4_2.env" "configs/wiki-convert_en_byte_e2e_true_0.0001_4_2.env" "configs/wiki-convert_en_char_e2e_true_0.0001_4_2.env" "configs/wiki-convert_en_sub_e2e_true_0.0001_4_2.env"

Submitting job with configs/wiki-convert_en_byte_no-e2e_true_0.0001_4_2.env
Submitted batch job 15329714

Submitting job with configs/wiki-convert_en_char_no-e2e_true_0.0001_4_2.env
Submitted batch job 15329715

Submitting job with configs/wiki-convert_en_sub_no-e2e_true_0.0001_4_2.env
Submitted batch job 15329716

Submitting job with configs/wiki-convert_en_word_no-e2e_true_0.0001_4_2.env
Submitted batch job 15329717

Submitting job with configs/wiki-convert_en_byte_e2e_true_0.0001_4_2.env
Submitted batch job 15329718

Submitting job with configs/wiki-convert_en_char_e2e_true_0.0001_4_2.env
Submitted batch job 15329719

Submitting job with configs/wiki-convert_en_sub_e2e_true_0.0001_4_2.env
Submitted batch job 15329720
