setup-travis:
	(sudo apt-cache show python3.6 | grep "Package: python3.6") || (sudo add-apt-repository ppa:deadsnakes/ppa -y; sudo apt update) || echo "0"
	sudo apt install python3 python3-pip pep8 python3-setuptools -y
	sudo pip3 install virtualenv
	virtualenv -p python3 venv

	bash -c 'source ./venv/bin/activate && pip install -r requirements.txt && deactivate'
	@echo "DONE!"

test:
	bash -c 'source ./venv/bin/activate && PYTHONPATH="./src/" pytest && deactivate'