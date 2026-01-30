


# '../experiment/1000_lora' is your config path, examples is correct


python pwc_prepare_data.py --work_dir  '../experiment/full_train_full_test/pwc_vanilla'
python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test/pwc_vanilla' --port 14527
python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test/pwc_vanilla' --batch_size 1

#python ../util/evaluate_ood.py --work_dir '../experiment/full_train_full_test/pwc_vanilla'
#python ../util/evaluate_iid.py --work_dir '../experiment/full_train_full_test/pwc_vanilla'
