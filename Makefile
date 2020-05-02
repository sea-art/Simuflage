setup-travis:
	sudo apt install python3 python3-pip -y
	sudo pip3 install virtualenv
	virtualenv -p python3 venv

	bash -c 'source ./venv/bin/activate && pip install -r requirements.txt && deactivate'

test:
	bash -c 'source ./venv/bin/activate && PYTHONPATH="./src/" pytest && deactivate'