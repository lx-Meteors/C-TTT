


# # # '../experiment/1000_lora' is your config path, examples is correct
# # #python ../util/evaluate_ood.py --work_dir '../experiment/full_train_full_test_15x/sac_pwc_lora_comp'
# # #python ../util/evaluate_iid.py --work_dir '../experiment/full_train_full_test_15x/sac_pwc_lora_comp'

# # # =========================================== SAC 15x ==============================================================
# # # sac_pwc_vanilla
# # python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_vanilla' --batch_size 1


# # # sac_pwc_lora_comp
# # python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_lora_comp' --port 14527
# # python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_lora_comp' --batch_size 1


# # # sac_pwc_comp
# # python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_comp' --port 14527
# # python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_comp' --batch_size 1


# # # sac_pwc_comp_test
# # python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_comp_test' --port 14527
# # python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/sac_pwc_comp_test' --batch_size 1

# # # pwc_vanilla_test
# # python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_15x/pwc_vanilla_test' --batch_size 1

# # # =========================================== SAC 5x ==============================================================

# # # sac_pwc_vanilla
# # python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_5x/sac_pwc_vanilla' --batch_size 1

# # # sac_pwc_comp
# # python ./instruction_trainer.py --work_dir   '../experiment/full_train_full_test_5x/sac_pwc_comp' --port 14527
# # python ./pwc_evaluator.py --work_dir   '../experiment/full_train_full_test_5x/sac_pwc_comp' --batch_size 1









# # =========================================== 500x 15x pwc ==============================================================
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/No-TTT/pwc/500x' --batch_size 1
# echo "[DONE] 500x 15x pwc task completed at $(date)"


# # =========================================== sac 15x pwc ==============================================================
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/No-TTT/pwc/SAC' --batch_size 1
# echo "[DONE] 15x sac pwc task completed at $(date)"


# # =========================================== vanilla 15x pwc ==============================================================
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/No-TTT/pwc/Vanilla' --batch_size 1
# echo "[DONE] 15x vanilla pwc task completed at $(date)"














# # =========================================== 500x 15x qusum ==============================================================
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/No-TTT/qusum/500x' --batch_size 1
# echo "[DONE] 500x 15x qusum task completed at $(date)"


# # =========================================== sac 15x qusum ==============================================================
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/No-TTT/qusum/SAC' --batch_size 1
# echo "[DONE] 15x sac qusum task completed at $(date)"


# =========================================== vanilla 15x qusum ==============================================================
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/No-TTT/qusum/Vanilla' --batch_size 1
echo "[DONE] 15x vanilla qusum task completed at $(date)"


# # =========================================== 500x 15x multifield ==============================================================
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/No-TTT/multifield/500x' --batch_size 1
# echo "[DONE] 500x 15x multifield task completed at $(date)"


# # =========================================== sac 15x multifield ==============================================================
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/No-TTT/multifield/SAC' --batch_size 1
# echo "[DONE] 15x sac multifield task completed at $(date)"


# # =========================================== vanilla 15x multifield ==============================================================
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/No-TTT/multifield/Vanilla' --batch_size 1
# echo "[DONE] 15x vanilla multifield task completed at $(date)"













# # =========================================== 500x ttt 15x pwc ==============================================================
# python ./instruction_trainer.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/pwc/500x' --port 14527
# echo "[DONE] 500x ttt 15x pwc task completed at $(date)"
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/pwc/500x' --batch_size 1
# echo "[DONE] 500x ttt 15x pwc task completed at $(date)"

# # =========================================== sac ttt 15x pwc ==============================================================
# python ./instruction_trainer.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/pwc/SAC' --port 14527
# echo "[DONE] sac ttt 15x pwc task completed at $(date)"
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/pwc/SAC' --batch_size 1
# echo "[DONE] sac ttt 15x pwc task completed at $(date)"


# # =========================================== 500x ttt 15x qusum ==============================================================
# python ./instruction_trainer.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/qusum/500x' --port 14527
# echo "[DONE] 500x ttt 15x qusum task completed at $(date)"
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/qusum/500x' --batch_size 1
# echo "[DONE] 500x ttt 15x qusum task completed at $(date)"


# # =========================================== sac ttt 15x qusum ==============================================================
# python ./instruction_trainer.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/qusum/SAC' --port 14527
# echo "[DONE] sac ttt 15x qusum task completed at $(date)"
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/qusum/SAC' --batch_size 1
# echo "[DONE] sac ttt 15x qusum task completed at $(date)"


# =========================================== 500x ttt 15x multifield ==============================================================
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_trainer.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/multifield/500x' --port 14527
# echo "[DONE] 500x ttt 15x multifield task completed at $(date)"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/multifield/500x' --batch_size 1
# echo "[DONE] 500x ttt 15x multifield task completed at $(date)"


# =========================================== sac ttt 15x multifield ==============================================================
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_trainer.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/multifield/SAC' --port 14527
# echo "[DONE] sac ttt 15x multifield task completed at $(date)"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/multifield/SAC' --batch_size 1
# echo "[DONE] sac ttt 15x multifield task completed at $(date)"




# # =========================================== 500x 15x multifield ==============================================================
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/pwc_test_qusum_mul/500x_qsum' --batch_size 1
# echo "[DONE] 500x 15x multifield task completed at $(date)"


# =========================================== sac 15x multifield ==============================================================
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/pwc_test_qusum_mul/500x_mul' --batch_size 1
# echo "[DONE] 15x sac multifield task completed at $(date)"


# # =========================================== 500x 15x multifield ==============================================================
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/pwc_test_qusum_mul/SAC_mul' --batch_size 1
# echo "[DONE] 500x 15x multifield task completed at $(date)"


# # =========================================== sac 15x multifield ==============================================================
# python ./instruction_evaluator.py --work_dir   '/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_15x/TTT/pwc_test_qusum_mul/SAC_qsum' --batch_size 1
# echo "[DONE] 15x sac multifield task completed at $(date)"



# nohup bash sft_script.sh > result.log 2>&1 &
# tail -f result.log
