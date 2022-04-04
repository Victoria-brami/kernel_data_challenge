# run with sh benchmark.sh > results/log_benchmark.txt
echo "Run default"
start_time=$(date +%s)
python predict.py --output_file "results/Yte_pred_default.csv"
end_time=$(date +%s)
echo "Runing time" $(( end_time - start_time )) "s"
echo "Run linear kernel"
start_time=$(date +%s)
python predict.py --kernel 'linear' --output_file "results/Yte_pred_klinear.csv"
end_time=$(date +%s)
echo "Runing time" $(( end_time - start_time )) "s"
echo "Run poly kernel"
start_time=$(date +%s)
python predict.py --kernel 'poly' --output_file "results/Yte_pred_kpoly.csv"
end_time=$(date +%s)
echo "Runing time" $(( end_time - start_time )) "s"
echo "Run ovo"
start_time=$(date +%s)
python predict.py --classifier_type 'ovo' --output_file "results/Yte_pred_ovo.csv"
end_time=$(date +%s)
echo "Runing time" $(( end_time - start_time )) "s"
echo "Run without hog"
start_time=$(date +%s)
python predict.py --feature_extractor None --output_file "results/Yte_pred_withouthog.csv"
end_time=$(date +%s)
echo "Runing time" $(( end_time - start_time )) "s"