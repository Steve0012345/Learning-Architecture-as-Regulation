#!/bin/bash
#
# usage: bash run_experiments.sh <main|ablation|platforms|grids|all> <data_folder> [or any other grid folders...]
#   e.g. bash run_experiments.sh main nyc_small/
#        bash run_experiments.sh grids nyc_small/ grid_small_dynamic/4/4/ grid_small_dynamic/8/8/ grid_small_dynamic/10/10/
#
# Every run trains for 10 episodes and is then tested on 5 held-out episodes. Results go in
# results/<scenario>/competitive_<architecture>/; pool them with: python summarize.py results/<scenario>/

STAGE=$1
DATA=$2
shift 2
EXTRA_GRIDS=("$@")

DEVICE=${DEVICE:-cpu}
EPOCHS=10
SEEDS=(5 8 47 94 106)
TD3="--pricing_alg TD3_MLP -alr 0.00001 -clr 0.001 --batch_size 32 -sa Gaussian -pd 0 -pe 64 -ac"
MARKET="--competitive --no_outside_option"

run () {
    local data=$1 seed=$2
    shift 2
    python main.py -d "$data" $TD3 $MARKET --n_epochs $EPOCHS --seed "$seed" -c "$DEVICE" -n "s$seed" "$@" -m all
    python main.py -d "$data" $TD3 $MARKET --n_epochs $EPOCHS --seed "$seed" -c "$DEVICE" -n "s$seed" "$@" -m test
}

main_comparison () {
    local data=$1
    shift
    for seed in "${SEEDS[@]}"; do
        run "$data" "$seed" --architecture local "$@"
        run "$data" "$seed" --architecture frl --lambda_rs 1.0 --tau_frl 0.5 "$@"
        run "$data" "$seed" --architecture hrl --lambda_rs 1.0 --hrl_interval 10 "$@"
        run "$data" "$seed" --architecture pfrl --lambda_rs 1.0 --tau_frl 0.5 --share_beta 0.4 "$@"
    done
}

ablations () {
    for seed in "${SEEDS[@]}"; do
        for lambda_rs in 0.5 1.5 2.0 3.0; do
            run "$DATA" "$seed" --architecture frl --lambda_rs $lambda_rs --tau_frl 0.5
        done
        for tau in 0.25 1.0 2.0; do
            run "$DATA" "$seed" --architecture frl --lambda_rs 1.0 --tau_frl $tau
        done
        for interval in 5 20; do
            run "$DATA" "$seed" --architecture hrl --lambda_rs 1.0 --hrl_interval $interval
        done
        for beta in 0.0 0.2 0.6 0.8 1.0; do
            run "$DATA" "$seed" --architecture pfrl --lambda_rs 1.0 --tau_frl 0.5 --share_beta $beta
        done
    done
}

platform_scaling () {
    # K = 2 is the main comparison, and total fleet and rider pool stay fixed as K changes
    for num_plat in 1 3 4; do
        main_comparison "$DATA" --num_platforms $num_plat
    done
}

grid_scaling () {
    for grid in "${EXTRA_GRIDS[@]}"; do
        main_comparison "$grid"
    done
}

case $STAGE in
    main) main_comparison "$DATA" ;;
    ablation) ablations ;;
    platforms) platform_scaling ;;
    grids) grid_scaling ;;
    all) main_comparison "$DATA"; ablations; platform_scaling; grid_scaling ;;
    *) echo "usage: bash run_experiments.sh <main|ablation|platforms|grids|all> <data_folder> [grid folders...]"; exit 1 ;;
esac
