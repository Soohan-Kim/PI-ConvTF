all: model eval

m_low = 0.9
m_high = 1.1
ttm_low = 0.04
ttm_high = 1.0
g = 20
conv_more_inputs = 0
model_name = all
model_save = Model1
p_low = 10
p_high = 90
include_more = 0
device= gpu:0

setup: requirements.txt 
	pip install -r requirements.txt

preprocess: preprocess.py
	python3 preprocess.py -m $(m_low) $(m_high) -ttm $(ttm_low) $(ttm_high) -g $(g) 

model: src
	cd src && python3 main.py -cmi $(conv_more_inputs) -model $(model_name) -save $(model_save) -device $(device)

eval: eval_src/call_eval.py eval_src/visualize.py results/MAPEs results/preds preprocess.py
	cd eval_src && python3 call_eval.py -p $(p_low) $(p_high) -save $(model_save)
	cd eval_src && python3 visualize.py -more $(include_more) -save $(model_save)
