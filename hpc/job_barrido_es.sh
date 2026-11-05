#!/bin/bash
#SBATCH -p docencia # cola
#SBATCH --cpus-per-task=32 # 125 ejecuciones independientes en paralelo
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --job-name=es_barrido
#SBATCH -o salida_barrido_%j.log

# Barrido ES: T0 x alpha (25 combinaciones x 5 semillas). Resultados en salidas/experimentos_es/
hostname
cd "$HOME/W/metaheuristicas" || exit 1
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
#conda activate metaheur
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
python -u experimentos_es.py barrido
