#!/bin/bash
TEST_SPEC=simple-c-package%clustcc-gcc@15.2.0
PORT_FILE=/tmp/port.txt
if [ -z $SPACK_ENV ]
then
    echo "Reactivating"
    spack env activate env/
fi
if [ ! -f $TEST_SPEC.spec.json ]
then
    echo "Concretizing $TEST_SPEC"
    spack spec --json $TEST_SPEC > $TEST_SPEC.spec.json
fi
echo "Removing old log files"
rm -rf logs/
echo "Cleaning cached downloads"
spack clean -d
echo "Ensuring no portfile"
rm -f $PORT_FILE
CLUSTCC_CMD="spack clustcc --logging-level debug --port-file $PORT_FILE"
mpirun -np 1 $CLUSTCC_CMD head \
       --spec-json $TEST_SPEC.spec.json \
       --local-concurrent-tasks 3 &
mpirun -np 4 $CLUSTCC_CMD worker &
wait
rm $PORT_FILE
