setup-travis:
	make setup
	@echo "DONE!"

setup:
	(sudo apt-cache show python3.6 | grep "Package: python3.6") || \
	(sudo add-apt-repository ppa:deadsnakes/ppa -y; sudo apt update) || echo "0"
	sudo apt install python3 python3-pip pep8 python3-setuptools -y
	sudo pip3 install virtualenv
	virtualenv -p python3 venv
	bash -c 'source ./venv/bin/activate && pip install -r requirements.txt && deactivate'

test:
	bash -c 'source ./venv/bin/activate && CODECOV_TOKEN="22f55b2c-7d73-4540-915a-7e6b61567e46" PYTHONPATH="./src/" coverage run -m pytest && deactivate'

check-pep8:
	bash -c 'source ./venv/bin/activate && pep8 --ignore E501 ./src/'

run:
	bash -c 'source ./venv/bin/activate && PYTHONPATH="./src/" python3 ./src/dse/montecarlo.py && deactivate'