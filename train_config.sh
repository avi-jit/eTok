for env_file in "$@"
do
    echo "Submitting job with $env_file"
    sbatch ./train.job -f $env_file
    echo ""
    sleep 1
done