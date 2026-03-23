#bin/bash
echo "1/12 - stp_base_big_0"
RUN_ID=stp_base_big_0 STP_LAMBDA=0.0 ITERATIONS=500 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_baseline.py
echo "2/12 - stp_base_small_0"
RUN_ID=stp_base_small_0 STP_LAMBDA=0.0 ITERATIONS=500 TRAIN_BATCH_TOKENS=16384 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_baseline.py
echo "3/12 - stp_base_big_02"
RUN_ID=stp_base_big_02 STP_LAMBDA=0.02 ITERATIONS=500 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_baseline.py
echo "4/12 - stp_base_small_02"
RUN_ID=stp_base_small_02 STP_LAMBDA=0.02 ITERATIONS=500 TRAIN_BATCH_TOKENS=16384 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_baseline.py
echo "5/12 - stp_base_big_06"
RUN_ID=stp_base_big_06 STP_LAMBDA=0.06 ITERATIONS=500 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_baseline.py
echo "6/12 - stp_base_small_06"
RUN_ID=stp_base_small_06 STP_LAMBDA=0.06 ITERATIONS=500 TRAIN_BATCH_TOKENS=16384 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_baseline.py
echo "7/12 - stp_sota_big_0"
RUN_ID=stp_sota_big_0 STP_LAMBDA=0.0 ITERATIONS=500 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 EVAL_STRIDE=0 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_sota.py
echo "8/12 - stp_sota_small_0"
RUN_ID=stp_sota_small_0 STP_LAMBDA=0.0 ITERATIONS=500 TRAIN_BATCH_TOKENS=16384 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 EVAL_STRIDE=0 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_sota.py
echo "9/12 - stp_sota_big_02"
RUN_ID=stp_sota_big_02 STP_LAMBDA=0.02 ITERATIONS=500 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 EVAL_STRIDE=0 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_sota.py
echo "10/12 - stp_sota_small_02"
RUN_ID=stp_sota_small_02 STP_LAMBDA=0.02 ITERATIONS=500 TRAIN_BATCH_TOKENS=16384 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 EVAL_STRIDE=0 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_sota.py
echo "11/12 - stp_sota_big_06"
RUN_ID=stp_sota_big_06 STP_LAMBDA=0.06 ITERATIONS=500 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 EVAL_STRIDE=0 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_sota.py
echo "12/12 - stp_sota_small_06"
RUN_ID=stp_sota_small_06 STP_LAMBDA=0.06 ITERATIONS=500 TRAIN_BATCH_TOKENS=16384 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 TTT_BATCH_SIZE=32 EVAL_STRIDE=0 PYTORCH_ALLOC_CONF=expandable_segments:True python3 stp/train_stp_sota.py
