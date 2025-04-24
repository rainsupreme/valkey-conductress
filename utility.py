import os
import subprocess
from numerize import numerize
from pathlib import Path
from config import sshkeyfile

million = 1000000
minute = 60
hour = 60*60

def human(number: float) -> str:
    return numerize.numerize(number)

def human_byte(number: float) -> str:
    number = float(number)
    units = ('B','KB','MB','GB','TB','PB')
    unit_index = 0
    while number >= 512 and unit_index+1 < len(units):
        number /= 1024
        unit_index += 1
    if number.is_integer():
        return f'{number:,g}{units[unit_index]}'
    else:
        return f'{number:,.1f}{units[unit_index]}'

def human_time(number: float) -> str:
    number = float(number)
    divisors = [1, 60, 60, 24]
    units = 'smhd'
    unit_index = 0
    while unit_index+1 < len(units) and number >= divisors[unit_index+1]:
        unit_index += 1
        number /= divisors[unit_index]
    if number.is_integer():
        return f'{number:,g}{units[unit_index]}'
    else:
        return f'{number:,.1f}{units[unit_index]}'

def pretty_header(text: str):
    margin = 1
    text = ' '*margin + text + ' '*margin

    endcap = '•'
    center = '•°•♥•°•'
    fill = '─'
    fillsize = len(text) - len(center) - len(endcap) * 2
    fillsize = fillsize//2 + 1
    divider = '\n' + endcap + fill*fillsize + center + fill*fillsize + endcap

    print(divider)
    print(text)

def calc_percentile_averages(data: list, percentages, lowestVals=False) -> tuple:
    copy = data.copy()
    copy.sort()
    if lowestVals:
        copy.reverse()

    result = []
    for percent in percentages:
        start_index = len(copy) * (100 - percent) // 100
        slice = copy[start_index:]
        slice_avg = sum(slice) / len(slice)
        result.append(slice_avg)
    return tuple(result)

def run_command(command: list, remote_ip=None, cwd=None, check=True):
    if remote_ip != None:
        command = ['ssh', '-i', sshkeyfile, remote_ip] + command
    result = subprocess.run(command, check=check, encoding='utf-8', cwd=cwd, stdout=subprocess.PIPE)
    return result.stdout

class RealtimeCommand:
    def __init__(self, command: list):
        self.p = subprocess.Popen(command, stdout=subprocess.PIPE)
        os.set_blocking(self.p.stdout.fileno(), False)
    def poll_output(self):
        output = self.p.stdout.read()
        if output != None:
            output = output.decode('utf-8')
        return output
    def is_running(self):
        return self.p.poll() == None
    def kill(self):
        self.p.kill()

def hash_local_file(path: Path):
    return run_command(['sha1sum', str(path)]).strip().split()[0]