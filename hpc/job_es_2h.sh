#!/bin/bash
#SBATCH -p long # cola
#SBATCH --cpus-per-task=16 
#SBATCH --time=24:00:00
#SBATCH --job-name=es_2h
#SBATCH -o salida_es2h_%j.log

# PON AQUÍ el T0 ganador del barrido (unidades absolutas de fitness)
T0=200
T_MIN=0.001

hostname
cd "$HOME/W/metaheuristicas" || exit 1
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
#conda activate metaheur
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
python -u run_es_2h.py --T0 $T0 --T_min $T_MIN --horas 2
