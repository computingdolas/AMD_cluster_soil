# Shoc level-0 Bechmarks

This folder is ready to run the shoc level 0 benchmarks for the HIP workshop.
This version is not guaranteed to work on other machines, and has been alterd to only run the level 0 benchmarks using hip.

A link to the shoc benchmark: [https://github.com/vetter/shoc/wiki]

## How to Run
1. cd ./shoc
2. sh config/conf-kleurplaat.sh
3. make
4. make install
5. ./bin/shocdriver -hip -s 4

config/conf-kleurplaat.sh calls the configure file, with some arguments required for compilation.
The shocdrive tool calls libexec/driver.pl which runs the benchmarks. Call ./bin/shocdriver --help for more driver options.
