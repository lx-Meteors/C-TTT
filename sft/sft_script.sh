


# '../experiment/1000_lora' is your config path, examples is correct
#python ../util/evaluate_ood.py --work_dir '../experiment/full_train_full_test_15x/sac_pwc_lora_comp'
#python ../util/evaluate_iid.py --work_dir '../experiment/full_train_full_test_15x/sac_pwc_lora_comp'

# =========================================== SAC 15x ==============================================================
# sac_pwc_vanilla
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_vanilla' --batch_size 1


# sac_pwc_lora_comp
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_lora_comp' --port 14527
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_lora_comp' --batch_size 1


# sac_pwc_comp
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_comp' --port 14527
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_comp' --batch_size 1


# sac_pwc_comp_test
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_comp_test' --port 14527
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_comp_test' --batch_size 1

# pwc_vanilla_test
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/pwc_vanilla_test' --batch_size 1

# =========================================== SAC 5x ==============================================================

# sac_pwc_vanilla
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_5x/sac_pwc_vanilla' --batch_size 1

# sac_pwc_comp
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_5x/sac_pwc_comp' --port 14527
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_5x/sac_pwc_comp' --batch_size 1




# =========================================== 500x 15x ==============================================================
# 500x_mrqa_vanilla
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/500x_mrqa_vanilla' --batch_size 1
python ../util/evaluate_ood.py --work_dir '../experiment/full_train_full_test_15x/500x_mrqa_vanilla'
python ../util/evaluate_iid.py --work_dir '../experiment/full_train_full_test_15x/500x_mrqa_vanilla'

# 500x_pwc_vanilla
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/500x_pwc_vanilla' --batch_size 1

# 500x_pwc_comp
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/500x_pwc_comp' --port 14527
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/500x_pwc_comp' --batch_size 1








