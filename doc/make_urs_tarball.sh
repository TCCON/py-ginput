#!/bin/bash
cd $(dirname $0)
cd build
rm -rf ginput-docs-html ginput-docs-html.tgz
cp -r html ginput-docs-html
cp ../README.md ginput-docs-html
tar -czf ginput-docs-html.tgz ginput-docs-html
