# CARC Cheatsheet

## Job Submissions

- `sbatch <job-file-name>.job`: Used to submit and queue your job for later execution (whenever resources are available).
- `salloc <resources-we-want>`: Used to request resources for internative use. (for complete list of resource flag, visit [CARC website](https://www.carc.usc.edu/user-information/user-guides/hpc-basics/slurm-cheatsheet#job-submission)

- SLURM Job Template file for ML

  ```bash
  #!/bin/bash
  #SBATCH --account=<account_number>
  #SBATCH --partition=gpu
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=4
  #SBATCH --mem=32GB
  #SBATCH --gres=gpu:a40:1
  #SBATCH --time=1:00:00
  #SBATCH --mail-user=<user_email>

  module purge
  module load gcc/11.3.0
  module load python/3.9.12
  module load cuda/11.6.2
  module load cudnn/8.4.0.27-11.6

  # Import evironment variables from .env file
  export $(grep -v '^#' .env | xargs -d '\n')

  if [ -d "venv" ]
  then
      echo "Found python virtual environment in working directory. Skipping creation ..."
  else
      echo "Python virtual environment not found. Creating virtual environment ..."
      python -m venv venv
  fi
  source venv/bin/activate

  pip install --upgrade -r requirements.txt
  python train.py -dataset $DATASET
  ```

- Available GPU resources on different partitions:

  - `--partition=gpu`

    - v100
    - p100
    - a40
    - a100

  - `--partition=debug`

    - k40
    - p100

  - `--partition=main`
    - k40

## Job Management

- `squeue`: View information about the jobs in the scheduling queue.

  - Commands that we might use the most:

    ```bash
    # View your own job queue with estimated start times
    squeue --me

    # View own job queue with estimated start times for pending jobs
    squeue --me --start
    ```

- `scancel <job-id>`: Cancel the specified job

## Monitoring

- `seff <job-id>`: Used to get the resource used by the job.
- `sacct -j <job-id> --format="JobID%20,JobName,User,AveDiskRead,AveDiskWrite,AveCPUFreq,Partition,NodeList,Elapsed,State,ExitCode,MaxRSS,AllocTRES%64,TRESUsageInAve%128"`: Used to get more detailed information about the job and the resources used by it during execution.
  (Note: All the values are in Bits)
- `nodeinfo -s idle,mix | grep gpu`: To list all the available partitions where we have GPUs available.
- `jobinfo <job-id>`: Compact resource summary of the given job.

FAQ / Common Issues

- CARC not able to ssh using Putty on windows systems.

- Remote SSH in VSCode

less slurm-13689062.out | grep acc_unit | tail
