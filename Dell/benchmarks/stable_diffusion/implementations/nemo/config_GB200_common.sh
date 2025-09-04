# Set the default distributed size for NVL72
gcd() ( ! (( $1 % $2 )) && echo $2 || gcd $2 $(( $1 % $2 )) )
export DISTRIBUTED_SIZE=${DISTRIBUTED_SIZE:-$((
    DGXNNODES * DGXNGPU <= 72 ?
        DGXNNODES * DGXNGPU :
        $( gcd 72 $((DGXNNODES * DGXNGPU)) )
))}
echo "DISTRIBUTED_SIZE=${DISTRIBUTED_SIZE}"
