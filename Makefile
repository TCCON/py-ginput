ENV ?= ginput-auto-default

help:
	@cat .makehelp

install: install-code install-man

install-code:
	./install.sh $(ENV)

install-man:
	$(MAKE) -C man install-man

run_ginput.py: .run_ginput_template.py
	./install-runscript.sh

fulltest:
	pytest -v tests/

test:
	pytest -v -m 'not glacial' tests/

quicktest:
	pytest -v -m 'not slow and not glacial' tests/

get-test-data:
	python -m ginput.testing.test_utils get

check-test-data:
	python -m ginput.testing.test_utils check

update-test-hashes:
	python -m ginput.testing.test_utils up

.PHONY: test quicktest test-profiles test-utils get-test-data check-test-data update-test-hashes
