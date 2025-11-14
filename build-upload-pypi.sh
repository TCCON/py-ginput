#!/bin/bash

if [[ $1 == "-h" ]] || [[ $1 == "--help" ]]; then
  echo "USAGE: $0 [[ -t | --test ]"
  echo "  The -t/--test flag will upload to testpypi instead"
  exit 0
fi

upload_args=""
if [[ $1 == "-t" ]] || [[ $1 == "--test" ]]; then
  upload_args="--repository testpypi"
fi

python3 -m pip install --upgrade build
python3 -m pip install --upgrade twine

rm -rfv dist/

python3 -m build

echo "If prompted for a username+password, use __token__ as the username and an API token as the password"

python3 -m twine upload dist/* --verbose $upload_args

echo "(It may not show up right away, so don't freak out if pip installing the new version fails initially.)"
