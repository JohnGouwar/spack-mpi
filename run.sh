#!/bin/bash
TEST_SPEC=zlib
LOG_FILE=head_node.log
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
    echo "Clean and GC"
    spack env deactivate
    spack clean -a
    spack gc -y
fi
if [ -z $SPACK_ENV ]
then
    echo "Reactivating"
    spack env activate env/
fi
if [ ! -z $LOG_FILE ]
then
    echo "Removing $LOG_FILE"
    rm -f $LOG_FILE
fi
mpirun -np 1 spack clustcc head \
       --spec-json $TEST_SPEC.spec.json \
       --clustcc-spec-json clustcc.spec.json \
       --local-concurrent-tasks 3 : -np 1 spack clustcc worker
