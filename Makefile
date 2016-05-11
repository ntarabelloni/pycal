init:
#	pip install -r requirements.txt

test:
	py.test tests

clean:
	find -name "*pyc" -delete
	find -name "__pycache__" -delete

