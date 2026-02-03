#!/usr/bin/env bash
to_tar=$(find . -maxdepth 1 | grep -v generate_tarball.sh | grep -v ginput-large-files.md5 | grep -v ginput-large-files.tgz | grep -v .gitignore | grep -v '^.$' | xargs echo)
echo "Will add the following to the tarball: $to_tar"
tar -czvf ginput-large-files.tgz $to_tar
find $to_tar -type f | xargs md5sum | tee ginput-large-files.md5
