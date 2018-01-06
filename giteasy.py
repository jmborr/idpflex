#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

from __future__ import print_function

import re
import subprocess
import argparse
import argcomplete


def retrieve_branch():
    proc = subprocess.Popen(['git', 'branch'], stdout=subprocess.PIPE)
    for line in proc.stdout:
        if '*' in line:
            return line.split()[1]


def suggest_action(error_message):
    r"""Suggest corrective action based on the output after running
    the Git commands"""
    current_branch = retrieve_branch()
    error_pattern = r"error: The branch.*is not fully merged[.\n]*If" \
                    r" you are sure you want to delete it"
    suggestion = "Checkout again to the branch you want do delete, and" \
                 " this time include flag --force"
    buffer = "TAKE ACTION: "
    if re.search(error_pattern, error_message):
        print(buffer + "You are in the "+current_branch+" branch. "+suggestion)


parser = argparse.ArgumentParser(description='Git commands for Mantid')
parser.add_argument('--create', type=str, default='',
                    help="Create a branch. The name should be"
                         " <git-issue>_descriptive_text")
parser.add_argument('--pushToOrigin', action='store_true',
                    help="push branch to origin")
parser.add_argument('--force', action='store_true',
                    help="in combination with pushToOrigin to overwrite"
                         " the remote branch.")
parser.add_argument('--update', action='store_true',
                    help="update branch with a master rebase.")
parser.add_argument('--delete', action='store_true',
                    help="delete current local branch")
parser.add_argument('--updateMaster', action='store_true',
                    help="update local master branch.")
parser.add_argument('--dryrun', action='store_true',
                    help="print commands to be run, but don't run them.")
argcomplete.autocomplete(parser)
args = parser.parse_args()

script = """#!/bin/bash
# COMMANDS to run
"""

if args.create:
    branch = args.create
    script += """git checkout master
git fetch -p
git branch --no-track {0} origin/master
git checkout {0}
""".format(branch)

if args.pushToOrigin:
    branch = retrieve_branch()
    if args.force:
        branch = "+" + branch
    if not branch or 'master' in branch:
        raise IOError("branch not retrieved")
    script += """git push origin {0}""".format(branch)

if args.update:
    script += """git fetch -p
git rebase -v origin/master"""

if args.delete:
    branch = retrieve_branch()
    script += """git checkout master
git branch -D {0}
""".format(branch)

if args.updateMaster:
    branch = retrieve_branch()
    if branch == 'master':
        script += """git fetch -p
git pull --rebase
"""

print(script)  # inform of the commands to be run
output = None  # collect output from the commands
if not args.dryrun:
    try:
        output = subprocess.check_output(script, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as errorObject:
        output = errorObject.output
    print("OUTPUT from the commands\n"+output)
    suggest_action(output)
