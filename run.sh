#!/bin/bash
TEST_SPEC=gmake
LOG_FILE=head_node.log
PORT_FILE=/tmp/port.txt
if [ ! -f $TEST_SPEC.spec.json ]
then
    echo "Concretizing $TEST_SPEC"
    spack spec --json $TEST_SPEC > $TEST_SPEC.spec.json
fi
if [ ! -f clustcc.spec.json ]
then
    echo "Concretizing wrapper"
    spack spec --json clustcc-compiler-wrapper > clustcc.spec.json;
fi
if [ ! -z $SPACK_ENV ]
then
    echo "Running GC"
    spack env deactivate
    spack gc -y
fi
if [ -z $SPACK_ENV ]
then
    echo "Reactivating"
    spack env activate env/
fi
if [ ! -z $LOG_FILE ]
then
    echo "Removing log files"
    rm -f *.log
fi
CLUSTCC_CMD="spack clustcc --logging-level debug --port-file $PORT_FILE"
mpirun -np 1 $CLUSTCC_CMD head \
       --spec-json $TEST_SPEC.spec.json \
       --clustcc-spec-json clustcc.spec.json \
       --local-concurrent-tasks 3 &
mpirun -np 4 $CLUSTCC_CMD worker &
wait
rm $PORT_FILE
