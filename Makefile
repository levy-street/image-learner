include .env
export


.venv:
	python3.12 -m venv .venv


install: .venv
	.venv/bin/pip install -r requirements.txt


run:
	.venv/bin/python main.py
