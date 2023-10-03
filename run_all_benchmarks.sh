#!/bin/bash

NUM_RUNS=10

# NONTOF sino projections
python 00_pet_sinogram_nontof.py --num_runs ${NUM_RUNS} --mode GPU --threadsperblock 32
python 00_pet_sinogram_nontof.py --num_runs ${NUM_RUNS} --mode GPU --threadsperblock 64
python 00_pet_sinogram_nontof.py --num_runs ${NUM_RUNS} --mode GPU --threadsperblock 16

python 00_pet_sinogram_nontof.py --num_runs ${NUM_RUNS} --mode hybrid --threadsperblock 32
python 00_pet_sinogram_nontof.py --num_runs ${NUM_RUNS} --mode hybrid --threadsperblock 64
python 00_pet_sinogram_nontof.py --num_runs ${NUM_RUNS} --mode hybrid --threadsperblock 16

python 00_pet_sinogram_nontof.py --num_runs ${NUM_RUNS} --mode CPU

# TOF sino projections
python 01_pet_sinogram_tof.py --num_runs ${NUM_RUNS} --mode GPU --threadsperblock 32
python 01_pet_sinogram_tof.py --num_runs ${NUM_RUNS} --mode GPU --threadsperblock 64
python 01_pet_sinogram_tof.py --num_runs ${NUM_RUNS} --mode GPU --threadsperblock 16

python 01_pet_sinogram_tof.py --num_runs ${NUM_RUNS} --mode hybrid --threadsperblock 32
python 01_pet_sinogram_tof.py --num_runs ${NUM_RUNS} --mode hybrid --threadsperblock 64
python 01_pet_sinogram_tof.py --num_runs ${NUM_RUNS} --mode hybrid --threadsperblock 16

python 01_pet_sinogram_tof.py --num_runs ${NUM_RUNS} --mode CPU

# LM projections
arr=( "40000000" "20000000" "10000000" "5000000" "2500000" "1250000" )
for NUM_EVENTS in "${arr[@]}"
do
  python 02_pet_lm_nontof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode GPU --threadsperblock 32
  python 02_pet_lm_nontof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode hybrid --threadsperblock 32
  python 02_pet_lm_nontof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode CPU --threadsperblock 32

  python 02_pet_lm_nontof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode GPU --threadsperblock 32 --presort
  python 02_pet_lm_nontof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode hybrid --threadsperblock 32 --presort
  python 02_pet_lm_nontof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode CPU --threadsperblock 32 --presort
  
  python 03_pet_lm_tof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode GPU --threadsperblock 32
  python 03_pet_lm_tof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode hybrid --threadsperblock 32
  python 03_pet_lm_tof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode CPU --threadsperblock 32

  python 03_pet_lm_tof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode GPU --threadsperblock 32 --presort
  python 03_pet_lm_tof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode hybrid --threadsperblock 32 --presort
  python 03_pet_lm_tof.py --num_runs ${NUM_RUNS} --num_events ${NUM_EVENTS} --mode CPU --threadsperblock 32 --presort

  for SA in 0 1 2 
  do
    python 04_pet_lm_tof_real_data.py --num_events ${NUM_EVENTS} --mode GPU --threadsperblock 32 --symmetry_axis ${SA}
    python 04_pet_lm_tof_real_data.py --num_events ${NUM_EVENTS} --mode hybrid --threadsperblock 32 --symmetry_axis ${SA}
    python 04_pet_lm_tof_real_data.py --num_events ${NUM_EVENTS} --mode CPU --threadsperblock 32 --symmetry_axis ${SA}

    python 04_pet_lm_tof_real_data.py --num_events ${NUM_EVENTS} --mode GPU --threadsperblock 32 --symmetry_axis ${SA} --presort
    python 04_pet_lm_tof_real_data.py --num_events ${NUM_EVENTS} --mode hybrid --threadsperblock 32 --symmetry_axis ${SA} --presort
    python 04_pet_lm_tof_real_data.py --num_events ${NUM_EVENTS} --mode CPU --threadsperblock 32 --symmetry_axis ${SA} --presort
  done
done
