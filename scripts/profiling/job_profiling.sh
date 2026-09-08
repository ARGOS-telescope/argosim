#!/bin/bash
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=NONE
#SBATCH --job-name=argosim_profiling    # nom du job
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --time=00:30:00               # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=out_argosim_profiling.out   # nom du fichier de sortie
#SBATCH --error=err_argosim_profiling.err    # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A prk@v100                   # specify the project
#SBATCH --qos=qos_gpu-dev             # qos_gpu-dev or qos_gpu-t3

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load anaconda-py3/2024.06
conda activate argos

# echo launched commands
set -x

cd ${WORK}/repos/argosim/scripts/profiling_conv

# Pin CPU threads so the CPU run is reproducible and matches the allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

# CPU-vs-GPU scaling study: FIX the image size (NPX) and field of view (FOV),
# and grow the TOTAL VISIBILITY COUNT (small / medium / big) via N_FREQS.
#   n_vis = n_baselines(702) * N_TIMES * N_FREQS
#   small : nf=16  -> ~4.0 M vis   (~1.5 GB)
#   medium: nf=64  -> ~16.2 M vis  (~3.6 GB)
#   big   : nf=128 -> ~32.3 M vis  (~6.5 GB)
# Peak memory is dominated by n_vis (~0.18 GB / million); NPX only adds a small
# grid term (~NPX^2); FOV does NOT affect memory. Push the 'big' N_FREQS higher
# (e.g. 256) once you confirm it fits the GPU allocation.
NPX=1024
FOV=1.0
TRACK_TIME=4.0
N_TIMES=360
OUTDIR=out_jz

# All three backends run in the SAME allocation (same node, CPUs and memory), so
# the only difference at each size is the backend: pure NumPy, JAX-CPU, JAX-GPU.
for N_FREQS in 16 64 128; do
    srun python profiling.py gpu   --npx ${NPX} --fov ${FOV} --track_time ${TRACK_TIME} --n_times ${N_TIMES} --n_freqs ${N_FREQS} --outdir ${OUTDIR}
    srun python profiling.py cpu   --npx ${NPX} --fov ${FOV} --track_time ${TRACK_TIME} --n_times ${N_TIMES} --n_freqs ${N_FREQS} --outdir ${OUTDIR}
    srun python profiling.py numpy --npx ${NPX} --fov ${FOV} --track_time ${TRACK_TIME} --n_times ${N_TIMES} --n_freqs ${N_FREQS} --outdir ${OUTDIR}
done