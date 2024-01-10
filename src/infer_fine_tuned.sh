if [ "${1}" == "" ]
then
    tasks=("sst2" "mnli")
else
    tasks=(${1})
fi
    

for task in ${tasks[@]}
do
    echo "Working on ${task} ..."
    
    dir_model="../models/${task}-bert-base-uncased-steps/"
    for ckpt in $(ls ${dir_model}/ | grep checkpoint)
    do
        step=$(echo ${ckpt} | cut -d '-' -f 2)
        if [ ${step} -gt 5500 ]
        then
            continue
        fi
        
        python ./train_glue.py \
               --path_dataset ../data/${task}/paraphrased.ds \
               --do_predict \
               --model_name_or_path "${dir_model}/${ckpt}" \
               --dataset "${task}"

        python ./train_glue.py \
               --path_dataset ../data/${task}/${task}-bert-base-uncased/paraphrased.ds \
               --do_predict \
               --model_name_or_path "${dir_model}/${ckpt}" \
               --dataset "${task}"
    
        python ./train_glue.py \
               --path_dataset ../data/${task}/${task}-bert-base-uncased/synonym/paraphrased.ds \
               --do_predict \
               --model_name_or_path "${dir_model}/${ckpt}" \
               --dataset "${task}"
    done

    for i in $(seq 1 5)
    do
        if [ "${i}" != "10" ]
        then
            ratio=0.${i}
        else
            ratio=1.0
        fi        
        dir_model="../models/${task}-bert-base-uncased-${ratio}"

        # python ./train_glue.py \
        #        --path_dataset ../data/${task}/${task}-bert-base-uncased/paraphrased.ds \
        #        --do_predict \
        #        --model_name_or_path "${dir_model}" \
        #        --dataset "${task}"
        
        # python ./train_glue.py \
        #        --path_dataset ../data/${task}/paraphrased.ds \
        #        --do_predict \
        #        --model_name_or_path "${dir_model}" \
        #        --dataset "${task}"

        # python ./train_glue.py \
        #        --path_dataset ../data/${task}/${task}-bert-base-uncased/synonym/paraphrased.ds \
        #        --do_predict \
        #        --model_name_or_path "${dir_model}" \
        #        --dataset "${task}"

    done

done

