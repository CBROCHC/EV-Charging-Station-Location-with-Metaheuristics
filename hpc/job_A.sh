#!/bin/bash
#SBATCH -p docencia # cola
##SBATCH --gres=gpu:1 # el AG solo usa CPU: no se pide GPU (no acapares)
#SBATCH --cpus-per-task=16 # nº de CPUs (máximo 96)
#SBATCH --time=23:00:00 # tiempo máximo (cola docencia: menos de 24 h)
#SBATCH --job-name=ag_A
#SBATCH -o salida_%j.log # log de salida

# FASE A: un factor del AG cada vez en la instancia de referencia (345 ejecuciones).
# Si el trabajo se corta, vuelve a lanzarlo con sbatch: solo se repiten las ejecuciones que faltan.

hostname # comandos a ejecutar
cd "$HOME/metaheuristicas" || exit 1
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
conda activate metaheur
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
python experimentos.py --fase A --workers $SLURM_CPUS_PER_TASK --salida resultados
python analisis.py --res resultados --out informe || echo "AVISO: analisis.py no pudo completarse"
bash empaquetar.sh A
