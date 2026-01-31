


# '../experiment/1000_lora' is your config path, examples is correct
#python ../util/evaluate_ood.py --work_dir '../experiment/full_train_full_test_15x/pwc_lora_comp'
#python ../util/evaluate_iid.py --work_dir '../experiment/full_train_full_test_15x/pwc_lora_comp'

# =========================================== 15x ==============================================================
# pwc_vanilla
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/pwc_vanilla' --batch_size 1


# pwc_lora_comp
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/pwc_lora_comp' --port 14527
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/pwc_lora_comp' --batch_size 1


# pwc_comp
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/pwc_comp' --port 14527
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/pwc_comp' --batch_size 1


# pwc_comp_test
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/pwc_comp_test' --port 14527
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/pwc_comp_test' --batch_size 1

# pwc_vanilla_test
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/pwc_vanilla_test' --batch_size 1

# =========================================== 5x ==============================================================

# pwc_vanilla
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_5x/pwc_vanilla' --batch_size 1

# pwc_comp
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_5x/pwc_comp' --port 14527
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_5x/pwc_comp' --batch_size 1