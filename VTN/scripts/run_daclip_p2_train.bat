@echo off
cd /d E:\restormer+volterra
python "E:\restormer+volterra\VTN\scripts\train_p2_daclip_baseline.py" --opt "E:\restormer+volterra\VTN\experiments\daclip_p2_train.yml" > "E:\restormer+volterra\VTN\results\train_daclip_p2_all_in_one\stdout.log" 2> "E:\restormer+volterra\VTN\results\train_daclip_p2_all_in_one\stderr.log"
