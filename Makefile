# Common commands for the team — run `make <target>`
# (or copy the underlying command if you're on Windows without make)

.PHONY: setup train evaluate demo clean

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

train:
	cd src && python train.py

evaluate:
	cd src && python evaluate.py

demo:
	streamlit run demo/app.py

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
