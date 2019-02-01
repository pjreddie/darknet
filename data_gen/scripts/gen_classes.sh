#!/usr/bin/env bash

# input Txx_date_unlabeled.zip or Txx_date.zip where xx are the tasks which the file will use to generate the class_lists

# This script file will auto generate the class_list.txt file for OpenLabeling using the name of the zip file and the class_numbers.txt file at the top YTS directory

echo $1
arg=$1

strindex() { 
  x="${1%%$2*}"
  [[ "$x" = "$1" ]] && echo -1 || echo "${#x}"
}

# Trim off all of the parts of the name except
search_char='_'
new_str=${arg%_*}
search_index=`strindex "$arg" "$search_char"`
if [ $search_index -ne -1 ]
then
    new_str=${new_str%_*}
fi
new_str=${new_str:1}
echo $new_str

# Okay the easiest way I can think to do this
# loop through each character in the string new_str
# if it's a normal number, append it to a task_arr
# if it's a ( skip it, set a flag go to the next character, store all following chars in a string, once I reach ) append that temp string to task_arr, unset the flag and keep looping
