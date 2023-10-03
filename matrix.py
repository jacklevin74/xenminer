#! /usr/bin/env python3

# Author: Joao S. O. Bueno
# gwidion@gmail.com
# GPL. v3.0

MAX_CASCADES = 1000
MAX_COLS = 20
FRAME_DELAY = 0.03

MAX_SPEED  = 8

import sqlite3, shutil, sys, time
from random import choice, randrange, paretovariate

CSI = "\x1b["
pr = lambda command: print("\x1b[", command, sep="", end="")
getchars = lambda start, end: [chr(i) for i in range(start, end)]

black, green, white = "30", "32", "37"

latin = getchars(0x30, 0x80)
greek = getchars(0x390, 0x3d0)
hebrew = getchars(0x5d0, 0x5eb)
cyrillic = getchars(0x400, 0x50)

chars= latin + greek + hebrew + cyrillic

# Define a larger, denser ASCII digit map for numbers 0-9
DIGIT_MAP = {
    '0': [
        "  ██████  ",
        " ███  ███ ",
        " ███  ███ ",
        " ███  ███ ",
        "  ██████  "
    ],
    '1': [
        "   ███    ",
        " ██████   ",
        "   ███    ",
        "   ███    ",
        " ███████  "
    ],
    '2': [
        " ███████  ",
        "███   ███ ",
        "    ███   ",
        "  ███     ",
        "█████████ "
    ],
    '3': [
        "████████  ",
        "      ███ ",
        " ███████  ",
        "      ███ ",
        "████████  "
    ],
    '4': [
        "███   ███ ",
        "███   ███ ",
        "█████████ ",
        "      ███ ",
        "      ███ "
    ],
    '5': [
        "█████████ ",
        "███       ",
        "████████  ",
        "      ███ ",
        "████████  "
    ],
    '6': [
        "  ███████ ",
        " ███      ",
        "████████  ",
        " ███  ███ ",
        "  ██████  "
    ],
    '7': [
        "█████████ ",
        "     ███  ",
        "    ███   ",
        "   ███    ",
        "  ███     "
    ],
    '8': [
        " ███████  ",
        "███   ███ ",
        " ███████  ",
        "███   ███ ",
        " ███████  "
    ],
    '9': [
        " ███████  ",
        "███   ███ ",
        " ████████ ",
        "      ███ ",
        " ███████  "
    ]
}

# function to print the large number
def print_large_number(number, x, y):
    number_str = str(number)
    lines = ["" for _ in range(5)]
    for digit in number_str:
        for i, line in enumerate(DIGIT_MAP[digit]):
            lines[i] += line + "  "
    for i, line in enumerate(lines):
        print_at(line, x - len(line) // 2, y - 2 + i, color=white, bright="1")


def get_latest_block_id():
    conn = sqlite3.connect('blocks.db')
    c = conn.cursor()
    c.execute("SELECT block_id FROM blocks ORDER BY block_id DESC LIMIT 1")
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    return None



def pareto(limit):
    scale = lines // 2
    number = (paretovariate(1.16) - 1) * scale
    return max(0, limit - number)

def init():
    global cols, lines
    cols, lines = shutil.get_terminal_size()
    pr("?25l")  # Hides cursor
    pr("s")  # Saves cursor position

def end():
    pr("m")   # reset attributes
    pr("2J")  # clear screen
    pr("u")  # Restores cursor position
    pr("?25h")  # Show cursor

def print_at(char, x, y, color="", bright="0"):
    pr("%d;%df" % (y, x))
    pr(bright + ";" + color + "m")
    print(char, end="", flush=True)

def update_line(speed, counter, line):
    counter += 1
    if counter >= speed:
        line += 1
        counter = 0
    return counter, line

def cascade(col):
    speed = randrange(1, MAX_SPEED)
    espeed = randrange(1, MAX_SPEED)
    line = counter = ecounter = 0
    oldline = eline = -1
    erasing = False
    bright = "1"
    limit = pareto(lines)
    while True:
        counter, line = update_line(speed , counter, line)
        if randrange(10 * speed) < 1:
            bright = "0"
        if line > 1 and line <= limit and oldline != line:
            print_at(choice(chars),col, line-1, green, bright)
        if line < limit:
            print_at(choice(chars),col, line, white, "1")
        if erasing:
            ecounter, eline = update_line(espeed, ecounter, eline)
            print_at(" ",col, eline, black)
        else:
            erasing = randrange(line + 1) > (lines / 2)
            eline = 0
        yield None
        oldline = line
        if eline >= limit:
            print_at(" ", col, oldline, black)
            break

def main():
    cascading = set()
    while True:
        # Update cascading characters first
        while add_new(cascading): pass
        stopped = iterate(cascading)
        sys.stdout.flush()
        cascading.difference_update(stopped)

        # Display the large block_id at the center
        latest_block_id = get_latest_block_id()
        if latest_block_id is not None:
            middle_line = lines // 2 - 2  # Adjusted for 5-row ASCII art
            middle_col = cols // 2 - len(str(latest_block_id)) * 2  # Adjusted for 4-col ASCII art per digit
            print_large_number(latest_block_id, middle_col, middle_line)
            
        time.sleep(FRAME_DELAY)

def add_new(cascading):
    if randrange(MAX_CASCADES + 1) > len(cascading):
        col = randrange(cols)
        middle_col = cols // 2
        skip_zone = 2  # define the zone around the center column where you skip falling letters
        # skip columns near the middle of the screen
        if middle_col - skip_zone < col < middle_col + skip_zone:
            return False
        for i in range(randrange(MAX_COLS)):
            cascading.add(cascade((col + i) % cols))
        return True
    return False

def iterate(cascading):
    stopped = set()
    for c in cascading:
        try:
            next(c)
        except StopIteration:
            stopped.add(c)
    return stopped

def doit():
    try:
        init()
        main()
    except KeyboardInterrupt:
        pass
    finally:
        end()

if __name__=="__main__":
    doit()
