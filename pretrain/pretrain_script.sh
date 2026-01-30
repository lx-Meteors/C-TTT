

# '../experiment/1000_lora' is your config path, examples is correct

python pre_prepare_data.py --work_dir '../experiment/8000_prompt_lora'
python ./pre_trainer.py --work_dir '../experiment/8000_lora' --port 14572
python ./pre_evaluator.py --work_dir '../experiment/1000_lora' --batch_size 1


