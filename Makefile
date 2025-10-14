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

test: test-profiles test-utils

quicktest: test-utils test-fo2

test-profiles: 
	python -m ginput.testing.mod_maker_tests
	
test-utils:
	python -m ginput.testing.utility_tests

test-fo2:
	python -m ginput.testing.fo2_tests

get-test-data:
	python -m ginput.testing.test_utils get

check-test-data:
	python -m ginput.testing.test_utils check

update-test-hashes:
	python -m ginput.testing.test_utils up

.PHONY: test quicktest test-profiles test-utils get-test-data check-test-data update-test-hashes
