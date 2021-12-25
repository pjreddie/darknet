import os
import configparser


if __name__ == "__main__":
    curr_path = os.getcwd()
    ls_files = os.listdir()

    # checking required files
    check = {}
    for file in ls_files:
        if ".data" in file:
            check['data'] = file

        if ".cfg" in file:
            check['cfg'] = file

        if ".names" in file:
            check['names'] = file

        if ".weights" in file:
            check['weights'] = file

    if len(check) != 4:
        print("missing files")
    else:
        print(check)

    # updating .data file
    with open('{}'.format(check['data']), 'r') as reader:
        lines = reader.readlines()

    lines[3] = "names={}/{}\n".format(curr_path, check['names'])
    with open('{}'.format(check['data']), 'w') as reader:
        for l in lines:
            reader.write(l)

    # creating ini file
    print("creating {}_cfg.ini".format(check['data'][:-5]))
    with open("{}_cfg.ini".format(check['data'][:-5]), 'w') as reader:
        for k, v in check.items():
            reader.write("[{}]\npath: {}/{}\n".format(k, curr_path, v))
            reader.write('\n')

    print("done")
