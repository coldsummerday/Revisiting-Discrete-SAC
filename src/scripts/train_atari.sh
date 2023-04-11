#!/bin/bash

if [ $# -gt 1 ]
then
    SEEDS=( $2 )
    ALG=$1
else
    if [ $# -gt 0 ]
    then
        SEEDS=( $1 )
        ALG="all"
    else
        SEEDS=( 1 )
        ALG="all"
    fi
fi

echo "SEED=$SEEDS"
echo "ALG=$ALG"

echo "SEED=$SEEDS"
TIMESTEP=1_000_000
TARGET_ENTROPY=0.5
BATCH_SIZE=64
BUFFER_SIZE=100_000
HIDDEN_DIM=512
NUM_LAYERS=2
LEARNING_RATE=3e-4

ENV_LIST=( "BankHeistNoFrameskip-v4" "AsterixNoFrameskip-v4" "QbertNoFrameskip-v4" "HeroNoFrameskip-v4" )

for SEED in "${SEEDS[@]}"; do
    for ENV in "${ENV_LIST[@]}"; do

        if [[ ("$ALG" = "all") || ("$ALG" = "sac") ]]
        then
			echo "Train SAC on $ENV"
            python -m examples.atari.atari_sac --task $ENV  \
			--total-timesteps $TIMESTEP --seed $SEED --batch-size $BATCH_SIZE \
			--logdir results --target-entropy-ratio $TARGET_ENTROPY \
			--hidden-size $HIDDEN_DIM --auto-alpha --alpha-lr $LEARNING_RATE \
			--actor-lr $LEARNING_RATE --critic-lr $LEARNING_RATE
        fi

        if [[ ("$ALG" = "all") || ("$ALG" = "sacreg") ]]
        then
			echo "Train regularized SAC on $ENV"
            python -m examples.atari.atari_sac --task $ENV  \
			--total-timesteps $TIMESTEP --seed $SEED --batch-size $BATCH_SIZE \
			--logdir results --target-entropy-ratio $TARGET_ENTROPY \
			--hidden-size $HIDDEN_DIM --auto-alpha --alpha-lr $LEARNING_RATE \
			--actor-lr $LEARNING_RATE --critic-lr $LEARNING_RATE \
			--regularized-softmax 
        fi
		
		if [[ ("$ALG" = "all") || ("$ALG" = "alpha_sac") ]]
        then
			echo "Train Clipping alpha SAC on $ENV"
            python -m examples.atari.atari_sac --task $ENV  \
			--total-timesteps $TIMESTEP --seed $SEED --batch-size $BATCH_SIZE \
			--logdir results --target-entropy-ratio $TARGET_ENTROPY \
			--hidden-size $HIDDEN_DIM --auto-alpha --alpha-lr $LEARNING_RATE \
			--actor-lr $LEARNING_RATE --critic-lr $LEARNING_RATE \
			--clip-alpha
        fi

		if [[ ("$ALG" = "all") || ("$ALG" = "revisit_sac") ]]
        then
			echo "Train revisit SAC on $ENV"
            python -m examples.atari.atari_sac --task $ENV  \
			--total-timesteps $TIMESTEP --seed $SEED --batch-size $BATCH_SIZE \
			--logdir results --target-entropy-ratio $TARGET_ENTROPY \
			--hidden-size $HIDDEN_DIM --auto-alpha --alpha-lr $LEARNING_RATE \
			--actor-lr $LEARNING_RATE --critic-lr $LEARNING_RATE \
			--avg-q --entropy-penalty --clip-q 
        fi
    done
done
