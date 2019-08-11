DATA_DIR=$(pwd)/data
echo "Create a folder $DATA_DIR"
mkdir ${DATA_DIR}

## DOWNLOAD GLUE DATA
## Please refer glue-baseline install requirments or other issues.
git clone https://github.com/jsalt18-sentence-repl/jiant.git
cd jiant
python scripts/download_glue_data.py --data_dir $DATA_DIR --tasks SST

cd ..
rm -rf jiant
#########################
