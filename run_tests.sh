#!/bin/bash

run_test() {
    local test_name=$1
    local command=$2
    echo -e "\n\033[1;34m▶️\033[0m Running ${test_name}..."
    ${command} > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "\033[1;32m✔\033[0m ${test_name} completed successfully."
    else
        echo -e "\033[1;31m✘\033[0m ${test_name} failed. Please check logs for details."
    fi
}
show_progress() {
    local current=$1
    local total=$2
    local progress=$(( (current * 100) / total ))
    local done=$(( (progress * 4) / 10 ))
    local left=$(( 40 - done ))
    local fill=$(printf "%${done}s")
    local empty=$(printf "%${left}s")
    printf "\rProgress: ${fill// /●}${empty// /○} ${progress}%%"
}
main() {
    local tests=(
        "AST HTS test:map -a ast -o hts -p E.coli_ATCC_25922_Hts -d tests/data_hts_ast.xlsx -e tests/test_result -m tests/map_hts.xlsx -r 2 -s 128 -f 1 -t 0.49"
        "AST Manual test:map -a ast -o manual -p E.coli_ATCC_25922_Manual -d tests/data_manual_ast.xlsx -e tests/test_result -m tests/map_manual.xlsx -r 2 -s 128 -f 0.25 -t 0.49"
        "HC50 HTS test:map -a hc50 -o hts -p RBC_Hts -d tests/data_hts_hc50.xlsx -e tests/test_result -m tests/map_hts.xlsx -r 2 -s 128 -f 1"
        "HC50 Manual test:map -a hc50 -o manual -p RBC_Manual -d tests/data_manual_hc50.xlsx -e tests/test_result -m tests/map_manual.xlsx -r 2 -s 128 -f 0.25"
        "CC50 Manual test:map -a cc50 -o manual -p HEK293_Manual -d tests/data_manual_cc50.xlsx -e tests/test_result -m tests/map_manual_cc50.xlsx -r 2 -s 128 -f 0.25"
    )
    local total_tests=${#tests[@]}
    for i in "${!tests[@]}"; do
        IFS=":" read -r test_name command <<< "${tests[$i]}"
        run_test "$test_name" "$command"
        show_progress $((i + 1)) $total_tests
    done
    echo -e "\n\033[1;31m*** Please comapre the test and expected results in 'tests' directory ***\033[0m"
}
main