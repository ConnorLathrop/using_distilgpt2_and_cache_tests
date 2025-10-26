# Simple script to just run each model after the other
if [ ! -d "venv" ]; then
    python -m venv venv;
    pip install -r requirements.txt;
fi
if [[ ! "$VIRTUAL_ENV" != "" ]]
then
  source venv/Scripts/activate
else
  echo "Already in venv"
fi
source venv/Scripts/activate
echo "Running all"
echo "Example"
python pretrained_model_run.py; 
echo "Experiment 1"
python kv_cache_run.py; 
echo "Experiment 2"
python batching_run.py; 
echo "Expierment 3"
python sequence_scaling_run.py