install-package:
	python3 -m pip install --upgrade pip
	pip install -e .
	pip install -r requirements.txt
	pip install --upgrade protobuf
    
download_data:
	python3 src/data_collection.py
