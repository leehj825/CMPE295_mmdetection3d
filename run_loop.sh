#!/bin/bash

#SBATCH --job-name=HJ_w
#SBATCH --output=log.loop2.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-00:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH --mail-user=hyejun.lee@sjsu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --partition=gpu

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
echo ':: Start ::'
source ~/.bashrc
# Number of iterations
NUM_ITERATIONS=5

for (( i=1; i<=NUM_ITERATIONS; i++ ))
do
    ITERATION_START_TIME=$(date +%s)  # Capture the start time of the iteration
    echo "Iteration $i start time: $(date -d @$ITERATION_START_TIME)"

    # Submit the first two jobs and store their IDs
    FIRST_JOB_IDS=()
    for j in 1 2
    do
        if [ $i -eq 10 ]; then
            SCRIPT_NAME="run_fcos3d_waymo_$j.sh"
        else
            SCRIPT_NAME="run_fcos3d_waymo_${j}_resume.sh"
        fi
        SBATCH_OUTPUT=$(sbatch --parsable $SCRIPT_NAME)
        JOB_ID=$(echo $SBATCH_OUTPUT | awk '{print $NF}')
        FIRST_JOB_IDS+=($JOB_ID)
        echo "Started job $JOB_ID with script $SCRIPT_NAME"
    done

    MINUTE_COUNTER_1_2=0
    # Wait for the first two jobs to complete
    for JOB_ID in ${FIRST_JOB_IDS[@]}
    do
        while squeue | grep -wq "$JOB_ID"; do
            if (( MINUTE_COUNTER_1_2 % 10 == 0 )); then
                CURRENT_TIME=$(date +%s)
                echo "Checked at $(date -d @$CURRENT_TIME): job $JOB_ID is still running..."
            fi
            sleep 60
            ((MINUTE_COUNTER_1_2++))
        done
    done

    echo "First two jobs have completed, starting the next two."
    SECOND_JOB_IDS=()
    # Submit the next two jobs with a dependency on the first two
    for j in 3 4
    do
        if [ $i -eq 10 ]; then
            SCRIPT_NAME="run_fcos3d_waymo_$j.sh"
        else
            SCRIPT_NAME="run_fcos3d_waymo_${j}_resume.sh"
        fi
        SBATCH_OUTPUT=$(sbatch --parsable $SCRIPT_NAME)
        JOB_ID=$(echo $SBATCH_OUTPUT | awk '{print $NF}')
        SECOND_JOB_IDS+=($JOB_ID)
        echo "Started job $JOB_ID with script $SCRIPT_NAME"
    done

    MINUTE_COUNTER_3_4=0
    # Wait for the first two jobs to complete
    for JOB_ID in ${SECOND_JOB_IDS[@]}
    do
        while squeue | grep -wq "$JOB_ID"; do
            if (( MINUTE_COUNTER_3_4 % 10 == 0 )); then
                CURRENT_TIME=$(date +%s)
                echo "Checked at $(date -d @$CURRENT_TIME): job $JOB_ID is still running..."
            fi
            sleep 60
            ((MINUTE_COUNTER_3_4++))
        done
    done

    # Now, all jobs have been submitted, we can move to the post-processing step
    # Wait for all jobs to complete before starting the post-processing job
    #JOB_IDS=($(squeue --name="run_fcos3d_waymo*" --format="%i" --noheader))
    #DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")

    # Submit the post-processing job
    SBATCH_OUTPUT=$(sbatch --parsable run_fedavg.sh)
    POST_PROCESS_JOB_ID=$(echo $SBATCH_OUTPUT | awk '{print $NF}')
    echo "Waiting for post-processing job $POST_PROCESS_JOB_ID to complete..."

    MINUTE_COUNTER=0
    while true; do
        if squeue | grep -wq "$POST_PROCESS_JOB_ID"; then
            if (( MINUTE_COUNTER % 10 == 0 )); then
                CURRENT_TIME=$(date +%s)
                echo "Checked at $(date -d @$CURRENT_TIME): Post-processing job is still running..."
            fi
            sleep 60
            ((MINUTE_COUNTER++))
        else
            CURRENT_TIME=$(date +%s)
            echo "At $(date -d @$CURRENT_TIME): Post-processing job has completed."
            break
        fi
    done
done

echo ':: End ::'
