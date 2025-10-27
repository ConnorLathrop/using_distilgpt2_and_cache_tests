# using_distilgpt2_and_cache_tests

This is a test of showing how caching and usage is affected by different parameters by using [distilgpt2](https://huggingface.co/distilbert/distilgpt2/blob/main/README.md?library=transformers)
The run_all.sh script will create a virtual enviornment if one doesn't already exist, install the requirements and then run each python file.

## Files
[pretrained_model_run.py](https://github.com/ConnorLathrop/using_distilgpt2_and_cache_tests/blob/main/pretrained_model_run.py)

This script loads the model tokenizes it and using a user prompt generates output. This is a basic test of the model to ensure everything works as expected

[kv_cache_run.py](https://github.com/ConnorLathrop/using_distilgpt2_and_cache_tests/blob/main/kv_cache_run.py)

This script uses the same model and generates outputs either using cache or not. This script prompts the user for the amount of tokens, and the logs in the outputs directory shows that as those tokens increase, the usefulness of the cache does as well.

[batching_run.py](https://github.com/ConnorLathrop/using_distilgpt2_and_cache_tests/blob/main/batching_run.py)

This script uses a different amount of prompts for each run and compares the amount of memory usage and utilization for the gpu while the program is running.

[sequence_scaling_run.py](https://github.com/ConnorLathrop/using_distilgpt2_and_cache_tests/blob/main/sequence_scaling_run.py)

This script generates input tokens and shows how response time and memory usage scales with the amount of input tokens provided. Plots are generated and saved to the outputs directory

## AI Usage
Some ai was used in generating and debugging of the code see `ai_responses` for those outputs
