#!/usr/bin/sh

rm -r documentation
mkdir documentation

pydoc -w ../ailib
pydoc -w ../ailib/*.py

mv *.html documentation/
