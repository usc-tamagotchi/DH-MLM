setting=${1:-"scratch"}
start=${2:-1}
end=${3:-10}


for i in $(seq ${start} ${end})
do
    if [ "${i}" != "10" ]
    then
        ratio=0.${i}
    else
        ratio=1.0
    fi
    echo $ratio

    case "${setting}" in
        "toys")
            extra_args=()
            ;;
        "toys-l0.5")
            extra_args=(--lang1_ratio 0.5)
            ;;
        "toys-l0.1")
            extra_args=(--lang1_ratio 0.1)
            ;;
        "toys-shift")
            extra_args=(--load_model ../models/random_walk/mlm-shift/)
            ;;
        "toys-shift")
            extra_args=(--load_model ../models/random_walk/mlm-shift/)
            ;;
        "toys-shift-l0.1")
            extra_args=(--load_model ../models/random_walk/mlm-shift/ --lang1_ratio 0.1)
            ;;
        "toys-shift-l0.5")
            extra_args=(--load_model ../models/random_walk/mlm-shift/ --lang1_ratio 0.5)
            ;;
        "toys-shift")
            extra_args=(--load_model ../models/random_walk/mlm-shift/)
            ;;
        "toys-shift-l0.1")
            extra_args=(--load_model ../models/random_walk/mlm-shift/ --lang1_ratio 0.1)
            ;;
        "toys-shift-l0.5")
            extra_args=(--load_model ../models/random_walk/mlm-shift/ --lang1_ratio 0.5)
            ;;
        "toys-mix")
            extra_args=(--load_model ../models/random_walk/mlm-mix/)
            ;;
        "toys-mix-l0.1")
            extra_args=(--load_model ../models/random_walk/mlm-mix/ --lang1_ratio 0.1)
            ;;
        "toys-mix-l0.5")
            extra_args=(--load_model ../models/random_walk/mlm-mix/ --lang1_ratio 0.5)
            ;;
        "toys-mix-shuffle")
            extra_args=(--load_model ../models/random_walk/mlm-mix/ --shuffle_weight)
            ;;
        "toys-w2v-freeze")
            extra_args=(--load_embedding ../models/random_walks/glove-mix/w2v.pt --freeze_emb)
            ;;
        "toys-w2v")
            extra_args=(--load_embedding ../models/random_walks/glove-mix/w2v.pt)
            ;;
        "toys-w2v-l0.1")
            extra_args=(--load_embedding ../models/random_walks/glove-mix/w2v.pt --lang1_ratio 0.1)
            ;;
        "toys-w2v-l0.5")
            extra_args=(--load_embedding ../models/random_walks/glove-mix/w2v.pt --lang1_ratio 0.5)
            ;;
        *)            
            exit;
    esac
    
    
    output_dir=../models/${setting}/${ratio}
    mkdir -p ${output_dir}
    srun --gres=gpu:1080:1 --time 10:00:00 \
         -o ${output_dir}/output.log \
         -e ${output_dir}/error.log \
         python ./train_toy.py --output_dir ${output_dir} \
         --train_ratio ${ratio} \
         --max_steps 30000 --per_device_train_batch_size 32 \
         --do_predict --do_train --eval_shift ${extra_args[@]} &

    
done
