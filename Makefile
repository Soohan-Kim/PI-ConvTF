
m_low = 0.9
m_high = 1.1
ttm_low = 0.04
ttm_high = 1
g = 20
conv_more_inputs = 0
fw = tf
model_name = all
model_save = Model1
p_low = 10
p_high = 90
include_more = 0
device= gpu:0

setup: tensorflow_src/requirements.txt pytorch_src/requirements.txt
ifeq ($(fw), tf)
	pip install -r tensorflow_src/requirements.txt
endif
ifeq ($(fw), torch)
	pip install -r pytorch_src/requirements.txt
endif

preprocess: preprocess.py
	python3 preprocess.py -m $(m_low) $(m_high) -ttm $(ttm_low) $(ttm_high) -g $(g) 

model: tensorflow_src pytorch_src get_data.py
ifeq ($(fw), tf)
	python3 tensorflow_src/main.py -m $(m_low) $(m_high) -ttm $(ttm_low) $(ttm_high) -g $(g) -save $(model_save) -device $(device)
endif
ifeq ($(fw), torch)
	python3 pytorch_src/main.py -m $(m_low) $(m_high) -ttm $(ttm_low) $(ttm_high) -g $(g) -cmi $(conv_more_inputs) -model $(model_name) -save $(model_save) -device $(device)
endif

eval: eval_src/call_eval.py eval_src/visualize.py results/MAPEs results/preds preprocess.py
	python3 eval_src/call_eval.py -p $(p_low) $(p_high) -save $(model_save)
	python3 eval_src/visualize.py -more $(include_more) -save $(model_save)
