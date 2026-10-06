#!/bin/bash
# A bash script to create tests for quantization

if [ -f zoo/speedup/script/sweeping_configs.yaml ]; then
    cp zoo/speedup/script/sweeping_configs.yaml zoo/speedup/script/sweeping_configs.yaml.backup
else
    echo "File not found!"
fi

echo "Creating bposd quantization..."
python zoo/speedup/script/update_quantization.py -i=2 -f=5 -d=bposd_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/bposd_surface_sweeping_h200_quant_25
python zoo/speedup/script/update_quantization.py -i=3 -f=4 -d=bposd_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/bposd_surface_sweeping_h200_quant_34
python zoo/speedup/script/update_quantization.py -i=4 -f=3 -d=bposd_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/bposd_surface_sweeping_h200_quant_43
python zoo/speedup/script/update_quantization.py -i=5 -f=2 -d=bposd_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/bposd_surface_sweeping_h200_quant_52


echo "Creating lottery_bp quantization..."
python zoo/speedup/script/update_quantization.py -i=2 -f=5 -d=lottery_bp_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/lottery_bp_surface_sweeping_h200_quant_25
python zoo/speedup/script/update_quantization.py -i=3 -f=4 -d=lottery_bp_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/lottery_bp_surface_sweeping_h200_quant_34
python zoo/speedup/script/update_quantization.py -i=4 -f=3 -d=lottery_bp_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/lottery_bp_surface_sweeping_h200_quant_43
python zoo/speedup/script/update_quantization.py -i=5 -f=2 -d=lottery_bp_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/lottery_bp_surface_sweeping_h200_quant_52


echo "Creating lottery_bposd quantization..."
python zoo/speedup/script/update_quantization.py -i=2 -f=5 -d=lottery_bposd_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/lottery_bposd_surface_sweeping_h200_quant_25
python zoo/speedup/script/update_quantization.py -i=3 -f=4 -d=lottery_bposd_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/lottery_bposd_surface_sweeping_h200_quant_34
python zoo/speedup/script/update_quantization.py -i=4 -f=3 -d=lottery_bposd_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/lottery_bposd_surface_sweeping_h200_quant_43
python zoo/speedup/script/update_quantization.py -i=5 -f=2 -d=lottery_bposd_quant
python zoo/speedup/script/generate_sweeping_configs.py -r=accuracy/lottery_bposd_surface_sweeping_h200_quant_52

reset to default qunatization
python zoo/speedup/script/update_quantization.py -i=3 -f=4

mv zoo/speedup/script/sweeping_configs.yaml.backup zoo/speedup/script/sweeping_configs.yaml

echo "All done."
