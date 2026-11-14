.PHONY: install test lint train evaluate visualize demo all clean

install:
	pip install -r requirements.txt

test:
	cd .. && python -m pytest BanglaProfanityDetection/tests/ -v

lint:
	python -m flake8 src/ app/ main.py --max-line-length 100 --ignore E501

train:
	python main.py --step train

evaluate:
	python main.py --step evaluate

visualize:
	python main.py --step visualize

demo:
	python main.py --step demo

all:
	python main.py --step all

clean:
	rm -rf results/*/run_*
	rm -rf results/*/frac_*
	rm -f plots/*.png
