#!/bin/bash
#SBATCH -p long # cola
#SBATCH --cpus-per-task=16 
#SBATCH --time=24:00:00
#SBATCH --job-name=ag_2h
#SBATCH -o salida_ag2h_%j.log

hostname
cd "$HOME/W/metaheuristicas" || exit 1
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
#conda activate metaheur
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
python -u run_ag_2h.py --horas 2 --poblacion 1000
